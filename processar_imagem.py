import os
import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from rasterio.windows import Window
from ultralytics import YOLO
from tqdm import tqdm 

print("Iniciando script de análise...")

MODEL_PATH = "bestv2.pt"
IMAGE_PATH = "../dados_brutos/Imagem_desafio.tif"
OUTPUT_GEOJSON = '../resultados_finais/deteccoes_processadas.geojson'

TILE_SIZE = 1024 # Tamanho usado no treinamento
OVERLAP = 150
CONFIDENCE_THRESHOLD = 0.60

# Cria a pasta de resultados, se não existir
os.makedirs('../resultados_finais', exist_ok=True)

print(f"Carregando modelo")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

print(f"Abrindo imagem")
try:
    with rasterio.open(IMAGE_PATH) as src:
        image_crs = src.crs
        image_transform = src.transform
        image_height = src.height
        image_width = src.width
        
        todas_deteccoes = []
        
        step = TILE_SIZE - OVERLAP
        
        # Calculando quantas fatias serão necessárias para usar no tqdm (opcional)
        total_tiles_y = int(np.ceil(image_height / step))
        total_tiles_x = int(np.ceil(image_width / step))
        total_tiles = total_tiles_y * total_tiles_x        
        
        with tqdm(total=total_tiles, desc="Processando Fatias") as pbar:
            for r_off in range(0, image_height, step):
                for c_off in range(0, image_width, step):
                    
                    window = Window(c_off, r_off, TILE_SIZE, TILE_SIZE)
                    

                    #Lê apenas as 3 primeiras bandas (RGB)
                    tile_data = src.read([1, 2, 3], window=window)
                    
                    # Converte de (Bandas, Altura, Largura) para (Altura, Largura, Bandas) para uso no YOLO
                    tile_rgb = np.transpose(tile_data, (1, 2, 0))
                    
                    results = model(tile_rgb, conf=CONFIDENCE_THRESHOLD, verbose=False)
                    
                    for r in results:
                        for box in r.boxes:
                            class_id = int(box.cls[0])
                            classe_nome = model.names[class_id]
                            confianca = float(box.conf[0])
                            
                            coords_cpu = box.xyxy[0].cpu()
                            x1_tile, y1_tile, x2_tile, y2_tile = coords_cpu
                            
                            # Calcula o centro da detecção
                            centro_x_tile = (x1_tile + x2_tile) / 2
                            centro_y_tile = (y1_tile + y2_tile) / 2
                            
                            # Transforma a coordenada do centro da detecção do TILE para a imagem inicial
                            x_global = c_off + centro_x_tile
                            y_global = r_off + centro_y_tile
                            
                            # Converte para coordenadas geográficas
                            lon, lat = src.xy(y_global, x_global)
                            
                            todas_deteccoes.append({
                                'classe': classe_nome,
                                'confianca': confianca,
                                'geometry': Point(lon, lat)
                            })
                    
                    pbar.update(1) # Atualiza a barra de progresso

        print("Varredura concluída.")

except Exception as e:
    print(f"Erro ao processar a imagem raster: {e}")
    exit()

if todas_deteccoes:
    
    gdf_bruto = gpd.GeoDataFrame(todas_deteccoes, crs=image_crs)
    
    gdf_bruto.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    
    print(f"Resultados salvos com sucesso em: {OUTPUT_GEOJSON}")
else:
    print("Nenhum objeto detectado na imagem.")

print("Script finalizado.")
