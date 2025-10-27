import geopandas as gpd
from shapely import Point
from shapely.geometry import Polygon
import numpy as np
import pyproj
import os
import pandas as pd
from scipy.spatial import cKDTree

print("Iniciando script: Filtrar por área e remover duplicatas...")

INPUT_GEOJSON = '../resultados_finais/deteccoes_processadas.geojson'
OUTPUT_FINAL = '../resultados_finais/eucaliptos_experimento_filtrados_deduplicados.geojson'
OUTPUT_GEOJSON_AREA_PLOT = '../resultados_finais/area_croqui_43x9_poligono.geojson' 
OUTPUT_GEOJSON_AREA_PLOT_BUFFERED = '../resultados_finais/area_croqui_43x9_poligono_buffered.geojson'
OUTPUT_EXCEL_CROQUI = '../resultados_finais/resultado_croqui.xlsx'

# Parâmetros do Experimento
PLOT_ROWS = 9
PLOT_COLS = 43
PLANT_SPACING_METERS = 2.0
ROW_SPACING_METERS = 3.2

# --- AZIMUTE ---
AZIMUTH_DEGREES = 97.856239

# Posição do Alvo
ALVO_REPRESENTA_LINHA = 1
ALVO_REPRESENTA_COLUNA = 1

# Distância mínima para considerar duplicatas
DUPLICATE_DISTANCE_METERS = 0.5 

PLOT_AREA_BUFFER_METERS = PLANT_SPACING_METERS / 2.0 

os.makedirs('../resultados_finais', exist_ok=True)

# Carrega as detecções processadas no script anterior
print(f"Carregando detecções processadas")
try:
    gdf_proc = gpd.read_file(INPUT_GEOJSON)
    if gdf_proc.empty:
        print("Arquivo de detecções processadas está vazio. Saindo.")
        exit()
    original_crs = gdf_proc.crs
    if original_crs is None:
         original_crs = "EPSG:4326"
         gdf_proc.crs = original_crs
    print(f"CRS Original das detecções: {original_crs}")
except Exception as e:
    print(f"Erro ao carregar o arquivo GeoJSON: {e}")
    exit()

# Encontra a posição do alvo
gdf_alvo = gdf_proc[gdf_proc['classe'] == 'alvo']
if gdf_alvo.empty:
    print("Nenhum 'alvo' encontrado.")
    exit()
elif len(gdf_alvo) > 1:
    print(f"{len(gdf_alvo)} alvos encontrados. Usando o primeiro.")
anchor_geom_orig_crs = gdf_alvo.geometry.iloc[0]

utm_crs = None
try:
    utm_crs = gdf_alvo.estimate_utm_crs()
    if utm_crs is None: raise ValueError("Não foi possível estimar UTM a partir do alvo.")
    print(f"CRS UTM estimado (a partir do alvo): {utm_crs}")
except Exception as e:
    print(f"AVISO: Falha ao estimar UTM a partir do alvo ({e}). Tentando a partir de todas as detecções...")
    try:
        utm_crs = gdf_proc.estimate_utm_crs()
        if utm_crs is None: raise ValueError("Não foi possível estimar UTM a partir de todas as detecções.")
        print(f"CRS UTM estimado (fallback - todas detecções): {utm_crs}")
    except Exception as e2:
        print(f"Erro fatal ao estimar CRS UTM: {e2}. Verifique geometrias.")
        exit()

# Define transformadores
transformer_to_utm = pyproj.Transformer.from_crs(original_crs, utm_crs, always_xy=True)
transformer_to_orig = pyproj.Transformer.from_crs(utm_crs, original_crs, always_xy=True)

# Coordenadas da âncora em Lat/Lon para geodésico
if original_crs.is_geographic:
    anchor_lon_orig, anchor_lat_orig = anchor_geom_orig_crs.x, anchor_geom_orig_crs.y
else:
    transformer_to_wgs84 = pyproj.Transformer.from_crs(original_crs, "EPSG:4326", always_xy=True)
    anchor_lon_orig, anchor_lat_orig = transformer_to_wgs84.transform(anchor_geom_orig_crs.x, anchor_geom_orig_crs.y)

geod = pyproj.Geod(ellps='WGS84') 

# Calcular área do experimento

def calculate_grid_point(start_lon, start_lat, target_row, target_col, 
                         anchor_row, anchor_col, 
                         plant_spacing, row_spacing, 
                         base_azimuth, perp_offset_angle=90):

    dist_along_row = (target_col - anchor_col) * plant_spacing 
    
    dist_between_rows = (target_row - anchor_row) * row_spacing
    
    lon1, lat1, _ = geod.fwd(start_lon, start_lat, base_azimuth, dist_along_row)
    lon_final, lat_final, _ = geod.fwd(lon1, lat1, base_azimuth + perp_offset_angle, dist_between_rows)
    return lon_final, lat_final

corner1_lon, corner1_lat = calculate_grid_point(anchor_lon_orig, anchor_lat_orig, 1, 1, ALVO_REPRESENTA_LINHA, ALVO_REPRESENTA_COLUNA, PLANT_SPACING_METERS, ROW_SPACING_METERS, AZIMUTH_DEGREES)
corner2_lon, corner2_lat = calculate_grid_point(anchor_lon_orig, anchor_lat_orig, 1, PLOT_COLS, ALVO_REPRESENTA_LINHA, ALVO_REPRESENTA_COLUNA, PLANT_SPACING_METERS, ROW_SPACING_METERS, AZIMUTH_DEGREES)
corner3_lon, corner3_lat = calculate_grid_point(anchor_lon_orig, anchor_lat_orig, PLOT_ROWS, PLOT_COLS, ALVO_REPRESENTA_LINHA, ALVO_REPRESENTA_COLUNA, PLANT_SPACING_METERS, ROW_SPACING_METERS, AZIMUTH_DEGREES)
corner4_lon, corner4_lat = calculate_grid_point(anchor_lon_orig, anchor_lat_orig, PLOT_ROWS, 1, ALVO_REPRESENTA_LINHA, ALVO_REPRESENTA_COLUNA, PLANT_SPACING_METERS, ROW_SPACING_METERS, AZIMUTH_DEGREES)

# Cria o polígono
plot_polygon_coords = [(corner1_lon, corner1_lat), (corner2_lon, corner2_lat), (corner3_lon, corner3_lat), (corner4_lon, corner4_lat), (corner1_lon, corner1_lat)]
plot_polygon = Polygon(plot_polygon_coords)
gdf_plot_area = gpd.GeoDataFrame([{'geometry': plot_polygon}], crs="EPSG:4326") # Cria em WGS84
final_filter_polygon = None 

try:

    #Converte para utm para aplicaro buffer e depois volta ao original
    gdf_plot_area_utm = gdf_plot_area.to_crs(utm_crs)
    plot_polygon_utm_buffered_geom = gdf_plot_area_utm.buffer(PLOT_AREA_BUFFER_METERS).iloc[0]
    gdf_plot_area_utm_buffered = gpd.GeoDataFrame([{'geometry': plot_polygon_utm_buffered_geom}], crs=utm_crs)
    gdf_plot_area_buffered_orig = gdf_plot_area_utm_buffered.to_crs(original_crs)
    final_filter_polygon = gdf_plot_area_buffered_orig.geometry.iloc[0]
    
except Exception as e:
    print(f"Erro ao criar/bufferizar polígono: {e}. Verifique coordenadas dos cantos.")
    exit()

# Filtra Eucaliptos dentro da área bufferizada

gdf_eucaliptos = gdf_proc[gdf_proc['classe'] == 'eucalipto'].copy()
gdf_eucaliptos_filtrados_area = gpd.GeoDataFrame(columns=gdf_eucaliptos.columns, crs=original_crs) 

if not gdf_eucaliptos.empty and final_filter_polygon is not None:
    if gdf_eucaliptos.crs != original_crs:
         gdf_eucaliptos = gdf_eucaliptos.to_crs(original_crs)
         
    is_within = gdf_eucaliptos.geometry.within(final_filter_polygon)
    gdf_eucaliptos_filtrados_area = gdf_eucaliptos[is_within].copy()
    num_filtrados_area = len(gdf_eucaliptos_filtrados_area)

    if num_filtrados_area == 0:
        print("Nenhum eucalipto detectado caiu dentro da área bufferizada.")
        exit()
elif gdf_eucaliptos.empty:
     print("Nenhum eucalipto detectado no arquivo de entrada.")
     exit()
else:
     print("Erro: Polígono de filtro.")
     exit()

# Remove pontos duplicados usando KDTree

gdf_euc_final = gpd.GeoDataFrame(columns=gdf_eucaliptos_filtrados_area.columns, crs=original_crs)

if not gdf_eucaliptos_filtrados_area.empty:

    try:
        gdf_euc_utm = gdf_eucaliptos_filtrados_area.to_crs(utm_crs)
    except Exception as e:
        print(f"Erro ao reprojetar para UTM: {e}.")
        gdf_euc_final = gdf_eucaliptos_filtrados_area # Pula a etapa
    else:
        if len(gdf_euc_utm) > 1:

            # Extrai coordenadas e confianças
            coords = np.array(list(gdf_euc_utm.geometry.apply(lambda p: (p.x, p.y))))
            confidences = gdf_euc_utm['confianca'].values
            
            # Constrói a KDTree
            tree = cKDTree(coords)
            
            pairs = tree.query_pairs(r=DUPLICATE_DISTANCE_METERS)
            
            indices_to_discard = set()
            
            for i, j in pairs:

                if i in indices_to_discard or j in indices_to_discard:
                    continue
                    
                # Compara as confianças dos dois pontos no par e descarta o de menor confiança
                if confidences[i] >= confidences[j]:
                    indices_to_discard.add(j)
                else:
                    indices_to_discard.add(i)
            
            keep_mask = np.ones(len(gdf_euc_utm), dtype=bool)
            if indices_to_discard: 
                 discard_list = list(indices_to_discard)
                 keep_mask[discard_list] = False

            # Seleciona os pontos finais e converte para o CRS original
            gdf_euc_utm_final = gdf_euc_utm[keep_mask].copy()
            gdf_euc_final = gdf_euc_utm_final.to_crs(original_crs)

            # Remove colunas temporárias, se existirem            
            cols_to_drop_final = ['buffer_geom', 'index_right', 'cluster_id']
            gdf_euc_final = gdf_euc_final.drop(columns=[col for col in cols_to_drop_final if col in gdf_euc_final.columns], errors='ignore')

        else: # Só tem 1 ponto
             print("Apenas 1 eucalipto na área, não é necessário deduplicar.")
             gdf_euc_final = gdf_eucaliptos_filtrados_area

else:
     print("Nenhum eucalipto estava dentro da área para deduplicar.")


if not gdf_euc_final.empty:
    print(f"\nSalvando resultado final ({len(gdf_euc_final)} eucaliptos filtrados e deduplicados)...")

    colunas_obrigatorias = ['classe', 'confianca', 'geometry']
    for col in colunas_obrigatorias:
        if col not in gdf_euc_final.columns:
            if col == 'confianca': gdf_euc_final[col] = 0.0 
            else: gdf_euc_final[col] = None 
    
    cols_to_save = ['classe', 'confianca', 'geometry'] 
    gdf_euc_final_limpo = gdf_euc_final[cols_to_save].copy() 

    try:
        gdf_euc_final_limpo.to_file(OUTPUT_FINAL, driver="GeoJSON")
    except Exception as e:
        print(f"Erro ao salvar GeoJSON final: {e}")
        try:
            gdf_euc_final_shp = gdf_euc_final_limpo.rename(columns={'confianca': 'Conf'}, errors='ignore')
            cols_shp = ['classe', 'Conf', 'geometry'] 
            gdf_euc_final_shp = gdf_euc_final_shp.reindex(columns=cols_shp, fill_value=None)
            gdf_euc_final_shp['Conf'] = gdf_euc_final_shp['Conf'].fillna(0.0)
            shp_path = OUTPUT_FINAL.replace('.geojson', '.shp')
            gdf_euc_final_shp.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')
            print(f"Resultado salvo como Shapefile em: {shp_path}")
        except Exception as e_shp:
            print(f"Erro ao salvar Shapefile: {e_shp}")

    print(f"\nGerando Excel no formato Croqui (Matriz) para: {OUTPUT_EXCEL_CROQUI}")
    try:
        planned_points = []
        for r in range(1, PLOT_ROWS + 1): #
            for c in range(1, PLOT_COLS + 1):
                lon, lat = calculate_grid_point(
                    anchor_lon_orig, anchor_lat_orig, r, c, 
                    ALVO_REPRESENTA_LINHA, ALVO_REPRESENTA_COLUNA,
                    PLANT_SPACING_METERS, ROW_SPACING_METERS,
                    AZIMUTH_DEGREES
                )
                planned_points.append({
                    'geometry': Point(lon, lat),
                    'Linha': r,
                    'Coluna': c
                })
        
        gdf_grid_planejado = gpd.GeoDataFrame(planned_points, crs="EPSG:4326")
        
        gdf_euc_final_utm = gdf_euc_final.to_crs(utm_crs)
        gdf_grid_planejado_utm = gdf_grid_planejado.to_crs(utm_crs)
        
        gdf_associado = gpd.sjoin_nearest(
            gdf_grid_planejado_utm, 
            gdf_euc_final_utm[['geometry', 'confianca']], 
            how='left', 
            max_distance = PLOT_AREA_BUFFER_METERS * 1.5, 
            rsuffix='detectado'
        )
        
        gdf_associado = gdf_associado.drop_duplicates(subset=['Linha', 'Coluna'], keep='first')
        
        pivot_table = gdf_associado.pivot_table(
            values='confianca',
            index='Linha',
            columns='Coluna',
            aggfunc='max'
        )
        
        pivot_table_sorted = pivot_table.sort_index(ascending=False)
        
        df_excel = pd.DataFrame(
            np.where(pivot_table_sorted.notnull(), "Encontrada", "Falha"),
            index=pivot_table_sorted.index,
            columns=pivot_table_sorted.columns
        )

        df_excel.to_excel(OUTPUT_EXCEL_CROQUI, index_label='Linha')

    except Exception as e:
        print(f"Erro ao gerar o Excel no formato croqui (matriz): {e}")

else:
    print("Nenhum eucalipto restou após todo o processamento.")

print("\nScript finalizado.")