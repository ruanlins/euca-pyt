import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import numpy as np
import pyproj
import os
from tqdm import tqdm # Import tqdm if not already there, for progress


# --- 1. CONFIGURAÇÕES ---

INPUT_GEOJSON_DETECTED = '../resultados_finais/eucaliptos_experimento_filtrados_deduplicados.geojson' # Saída do script anterior
INPUT_GEOJSON_ALVO = '../resultados_finais/deteccoes_processadas.geojson' # Para pegar o alvo
OUTPUT_GEOJSON_INFERIDOS = '../resultados_finais/eucaliptos_inferidos_ausentes.geojson'
OUTPUT_GEOJSON_GRADE_TEORICA = '../resultados_finais/grade_teorica_completa.geojson' # Salvar a grade para visualização

# Parâmetros do Croqui/Experimento
PLOT_ROWS = 9 # Número de linhas
PLOT_COLS = 43  # Número de colunas 
PLANT_SPACING_METERS = 2.0 # Espaçamento entre plantas na mesma linha (COLUNAS)
ROW_SPACING_METERS = 3.2 # Espaçamento entre linhas (LINHAS)

# --- AZIMUTE ---
# Azimute das linhas de plantio em graus (0=Norte, 90=Leste, 180=Sul, 270=Oeste)
AZIMUTH_DEGREES = 97.856239 # <--- VALOR MEDIDO NO QGIS

# Posição do Alvo no Grid (Âncora)
ALVO_REPRESENTA_LINHA = 1 
ALVO_REPRESENTA_COLUNA = 1

# Tolerância para associar detectados à grade teórica (metros)
# Ajuste se a associação estiver muito baixa (verifique alinhamento no QGIS primeiro)
ASSOCIATION_TOLERANCE_METERS = 2.0 # PLANT_SPACING_METERS 

# Limiar de distância para considerar vizinhos válidos para inferência (metros)
MAX_VALID_NEIGHBOR_DIST_FACTOR = 1.5 # Multiplicador sobre o espaçamento esperado
MAX_VALID_NEIGHBOR_DIST = PLANT_SPACING_METERS * MAX_VALID_NEIGHBOR_DIST_FACTOR 

# Garante que a pasta de resultados exista
os.makedirs('../resultados_finais', exist_ok=True)


# --- 2. CARREGAR DADOS ---

print(f"Carregando eucaliptos detectados de: {INPUT_GEOJSON_DETECTED}")
try:
    gdf_detected = gpd.read_file(INPUT_GEOJSON_DETECTED)
    if gdf_detected.empty:
        print("Arquivo de eucaliptos detectados está vazio. Nada a inferir. Saindo.")
        exit()
    original_crs = gdf_detected.crs
    if original_crs is None:
         original_crs = "EPSG:4326" # Assume WGS84
         gdf_detected.crs = original_crs
    print(f"CRS Original detectados: {original_crs}")
except Exception as e:
    print(f"Erro ao carregar o arquivo GeoJSON de detectados: {e}")
    exit()

print(f"Carregando alvo de: {INPUT_GEOJSON_ALVO}")
try:
    gdf_proc = gpd.read_file(INPUT_GEOJSON_ALVO)
    gdf_alvo = gdf_proc[gdf_proc['classe'] == 'alvo']
    if gdf_alvo.empty:
        print("ERRO CRÍTICO: Nenhum 'alvo' encontrado. Não é possível gerar grade.")
        exit()
    # Garante que o alvo use o mesmo CRS dos detectados antes de prosseguir
    if gdf_alvo.crs is None: gdf_alvo.crs = original_crs
    if gdf_alvo.crs != original_crs: gdf_alvo = gdf_alvo.to_crs(original_crs)
    
    anchor_geom_orig_crs = gdf_alvo.geometry.iloc[0] 
    print(f"Alvo (âncora) encontrado.")
except Exception as e:
     print(f"Erro ao carregar ou encontrar o alvo: {e}")
     exit()

# --- 3. PREPARAR PARA CÁLCULOS ---

utm_crs = None
try:
    utm_crs = gdf_alvo.estimate_utm_crs() 
    if utm_crs is None: raise ValueError("Estimativa UTM falhou.")
    print(f"CRS UTM estimado: {utm_crs}")
except Exception as e:
    print(f"Erro fatal ao estimar CRS UTM: {e}.")
    exit()

transformer_to_utm = pyproj.Transformer.from_crs(original_crs, utm_crs, always_xy=True)
transformer_to_orig = pyproj.Transformer.from_crs(utm_crs, original_crs, always_xy=True)
geod = pyproj.Geod(ellps='WGS84') 

# Coordenadas da âncora em Lat/Lon para geodésico
if original_crs.is_geographic:
    anchor_lon_orig, anchor_lat_orig = anchor_geom_orig_crs.x, anchor_geom_orig_crs.y
else:
    transformer_to_wgs84 = pyproj.Transformer.from_crs(original_crs, "EPSG:4326", always_xy=True)
    anchor_lon_orig, anchor_lat_orig = transformer_to_wgs84.transform(anchor_geom_orig_crs.x, anchor_geom_orig_crs.y)

# Função auxiliar
def calculate_grid_point(start_lon, start_lat, target_row, target_col, 
                         anchor_row, anchor_col, 
                         plant_spacing, row_spacing, 
                         base_azimuth, perp_offset_angle=90):
    dist_along_row = (target_col - anchor_col) * plant_spacing 
    dist_between_rows = (target_row - anchor_row) * row_spacing 
    lon1, lat1, _ = geod.fwd(start_lon, start_lat, base_azimuth, dist_along_row)
    lon_final, lat_final, _ = geod.fwd(lon1, lat1, base_azimuth + perp_offset_angle, dist_between_rows)
    return lon_final, lat_final

# --- 4. GERAR GRADE TEÓRICA COMPLETA --- 

print("Gerando grade teórica completa para associação...")
pontos_teoricos = []
perp_offset = 90 # Ajuste se necessário (+90 ou -90 para direção N/S)
for r in tqdm(range(1, PLOT_ROWS + 1), desc="Gerando Linhas Teóricas"):
    for c in range(1, PLOT_COLS + 1):
        lon, lat = calculate_grid_point(anchor_lon_orig, anchor_lat_orig, r, c, 
                                        ALVO_REPRESENTA_LINHA, ALVO_REPRESENTA_COLUNA, 
                                        PLANT_SPACING_METERS, ROW_SPACING_METERS, 
                                        AZIMUTH_DEGREES, perp_offset)
        pontos_teoricos.append({'Linha': r, 'Coluna': c, 'ID': f"P{r:02d}{c:02d}", 'LC': f"L{r}C{c}", 'geometry': Point(lon, lat)})

gdf_teorico = gpd.GeoDataFrame(pontos_teoricos, crs=original_crs) 
print(f"Grade teórica com {len(gdf_teorico)} pontos criada.")

# Salva a grade teórica para visualização
try:
    gdf_teorico.to_file(OUTPUT_GEOJSON_GRADE_TEORICA, driver="GeoJSON")
    print(f"Grade teórica salva em: {OUTPUT_GEOJSON_GRADE_TEORICA}")
except Exception as e:
    print(f"Erro ao salvar grade teórica: {e}")

# --- 5. ASSOCIAR PONTOS DETECTADOS À GRADE TEÓRICA ---

print("Associando eucaliptos detectados à grade teórica...")
# Converte ambos para UTM para join por distância métrica
gdf_detected_utm = gdf_detected.to_crs(utm_crs)
gdf_teorico_utm = gdf_teorico.to_crs(utm_crs)

# Para cada ponto DETECTADO, encontra o ponto TEÓRICO mais próximo
gdf_detected_matched = gpd.sjoin_nearest(
    gdf_detected_utm, # Detectados à esquerda
    gdf_teorico_utm[['Linha', 'Coluna', 'geometry']], # Teóricos à direita
    how='left', 
    max_distance=ASSOCIATION_TOLERANCE_METERS, 
    distance_col='dist_to_teorico' 
)

original_count = len(gdf_detected_matched)
gdf_detected_matched = gdf_detected_matched.dropna(subset=['index_right']) # Remove os sem match
matched_count = len(gdf_detected_matched)
print(f"Associados {matched_count} de {original_count} eucaliptos detectados a posições na grade.")

if matched_count == 0:
    print("Nenhum eucalipto detectado pôde ser associado à grade. Não é possível inferir. Saindo.")
    exit()

# Ordena por Linha e Coluna
gdf_detected_matched = gdf_detected_matched.sort_values(by=['Linha', 'Coluna'])

# --- 6. ITERAR POR LINHA E INFERIR AUSENTES ---

print("Iniciando inferência de posições ausentes...")
pontos_inferidos = []

for linha_num in tqdm(range(1, PLOT_ROWS + 1), desc="Processando Linhas"):
    gdf_linha_detectada = gdf_detected_matched[gdf_detected_matched['Linha'] == linha_num]
    num_detectados_linha = len(gdf_linha_detectada)
    gdf_linha_teorica = gdf_teorico[gdf_teorico['Linha'] == linha_num].sort_values(by='Coluna')
    
    if num_detectados_linha == PLOT_COLS: continue 
    if num_detectados_linha == 0: continue # Não infere se a linha está totalmente vazia

    # print(f"Linha {linha_num}: Encontrados {num_detectados_linha}/{PLOT_COLS}. Inferindo {PLOT_COLS - num_detectados_linha} posições...") # Log opcional
    
    colunas_teoricas = set(range(1, PLOT_COLS + 1))
    colunas_detectadas = set(gdf_linha_detectada['Coluna'].unique())
    colunas_ausentes = sorted(list(colunas_teoricas - colunas_detectadas))
    
    # Usa GDF já em UTM e ordenado
    gdf_linha_detectada_utm = gdf_linha_detectada # Já está em UTM e ordenado
    
    for col_ausente in colunas_ausentes:
        vizinho_antes = gdf_linha_detectada_utm[gdf_linha_detectada_utm['Coluna'] < col_ausente].iloc[-1:] 
        vizinho_depois = gdf_linha_detectada_utm[gdf_linha_detectada_utm['Coluna'] > col_ausente].iloc[0:1]

        ponto_referencia_lon = None
        ponto_referencia_lat = None
        azimute_inferencia = AZIMUTH_DEGREES # Padrão
        distancia_base = 0
        espaco_inferido = PLANT_SPACING_METERS # Padrão

        # Caso 1: Tem vizinhos dos dois lados
        if not vizinho_antes.empty and not vizinho_depois.empty:
            ponto_antes_utm = vizinho_antes.geometry.iloc[0]
            ponto_depois_utm = vizinho_depois.geometry.iloc[0]
            col_antes = vizinho_antes['Coluna'].iloc[0]
            col_depois = vizinho_depois['Coluna'].iloc[0]

            # --- CORREÇÃO geod.inv ---
            # Converte coordenadas UTM dos vizinhos para Lon/Lat (original_crs)
            lon_antes, lat_antes = transformer_to_orig.transform(ponto_antes_utm.x, ponto_antes_utm.y)
            lon_depois, lat_depois = transformer_to_orig.transform(ponto_depois_utm.x, ponto_depois_utm.y)
            
            # Calcula azimute e distância REAL usando Lon/Lat
            dist_vizinhos, az_vizinhos, _ = geod.inv(lon_antes, lat_antes, 
                                                     lon_depois, lat_depois, 
                                                     radians=False) # Azimute em graus
            # --- FIM CORREÇÃO ---
                                                      
            num_esperados_entre = col_depois - col_antes
            
            # Verifica se a distância real é razoável
            # abs() trata azimutes próximos de 0/360
            if abs(dist_vizinhos - num_esperados_entre * PLANT_SPACING_METERS) < MAX_VALID_NEIGHBOR_DIST and num_esperados_entre > 0:
                 azimute_inferencia = az_vizinhos 
                 espaco_inferido = dist_vizinhos / num_esperados_entre 
            else:
                 # print(f"  Aviso L{linha_num}C{col_ausente}: Dist ({dist_vizinhos:.1f}m) entre C{col_antes} e C{col_depois} inválida. Usando azimute global.") # Log Opcional
                 espaco_inferido = PLANT_SPACING_METERS 

            # Define ponto de referência (o anterior) e distância
            ponto_ref_utm = ponto_antes_utm
            distancia_base = (col_ausente - col_antes) * espaco_inferido
            ponto_referencia_lon, ponto_referencia_lat = lon_antes, lat_antes # Já temos em Lon/Lat

        # Caso 2: Só tem vizinho antes
        elif not vizinho_antes.empty:
            ponto_ref_utm = vizinho_antes.geometry.iloc[0]
            col_antes = vizinho_antes['Coluna'].iloc[0]
            distancia_base = (col_ausente - col_antes) * PLANT_SPACING_METERS 
            azimute_inferencia = AZIMUTH_DEGREES 
            ponto_referencia_lon, ponto_referencia_lat = transformer_to_orig.transform(ponto_ref_utm.x, ponto_ref_utm.y)

        # Caso 3: Só tem vizinho depois
        elif not vizinho_depois.empty:
            ponto_ref_utm = vizinho_depois.geometry.iloc[0]
            col_depois = vizinho_depois['Coluna'].iloc[0]
            distancia_base = (col_ausente - col_depois) * PLANT_SPACING_METERS # Negativa
            azimute_inferencia = AZIMUTH_DEGREES 
            ponto_referencia_lon, ponto_referencia_lat = transformer_to_orig.transform(ponto_ref_utm.x, ponto_ref_utm.y)
            
        # Calcula a posição inferida se tiver referência
        if ponto_referencia_lon is not None:
            lon_inferido, lat_inferido, _ = geod.fwd(ponto_referencia_lon, ponto_referencia_lat, 
                                                     azimute_inferencia, distancia_base)
            
            pontos_inferidos.append({
                'Linha': linha_num,
                'Coluna': col_ausente,
                'LC': f"L{linha_num}C{col_ausente}",
                'Observacao': 'Inferido',
                'geometry': Point(lon_inferido, lat_inferido)
            })
        # else: # Log Opcional
             # print(f"  Aviso L{linha_num}C{col_ausente}: Sem referência para inferência.")

# --- 7. CRIAR E SALVAR GEOJSON DOS PONTOS INFERIDOS ---

print(f"\nTotal de {len(pontos_inferidos)} posições inferidas.")
if pontos_inferidos:
    gdf_inferidos = gpd.GeoDataFrame(pontos_inferidos, crs=original_crs) 
    
    print(f"Salvando pontos inferidos em: {OUTPUT_GEOJSON_INFERIDOS}")
    try:
        gdf_inferidos.to_file(OUTPUT_GEOJSON_INFERIDOS, driver="GeoJSON")
        print("Arquivo GeoJSON de pontos inferidos salvo.")
    except Exception as e:
        print(f"Erro ao salvar GeoJSON de inferidos: {e}. Tentando Shapefile...")
        try:
             gdf_inferidos_shp = gdf_inferidos.copy()
             shp_path = OUTPUT_GEOJSON_INFERIDOS.replace('.geojson', '.shp')
             # Simplificar colunas para shapefile se necessário
             gdf_inferidos_shp[['Linha', 'Coluna', 'LC', 'Observacao', 'geometry']].to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')
             print(f"Resultado salvo como Shapefile em: {shp_path}")
        except Exception as e_shp:
             print(f"Erro ao salvar Shapefile de inferidos: {e_shp}")
else:
    print("Nenhuma posição foi inferida.")

print("\nScript finalizado.")