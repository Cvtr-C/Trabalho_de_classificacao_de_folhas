import cv2
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import label, regionprops
import numpy as np

j = 0
dados_completos = []
dado = pd.read_csv("Leaves/all.csv")
niveis_lbp = np.arange(256)
for _, linhas in dado.iterrows():
    # Imagem
    nome = linhas["id"]
    classe = linhas["y"]
    img = cv2.imread("Leaves/" + nome, 0)
    j += 1
    if img is None:
        print(f"Erro ao carregar a imagem {nome}")
        continue
    print(f"Carregando imagem ({j}/1907)", end="\r", flush=True)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # Limiarização OTSU
    _, binario = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Imagem segmentada
    img_segmentada = cv2.bitwise_and(img, img, mask=binario)
    # GLCM
    glcm = graycomatrix(
        img_segmentada,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )
    dissimilaridade = graycoprops(glcm, "dissimilarity")[0, 0]
    correlação = graycoprops(glcm, "correlation")[0, 0]
    homogeneidade = graycoprops(glcm, "homogeneity")[0, 0]
    asm = graycoprops(glcm, "ASM")[0, 0]
    energia = np.sqrt(asm)
    # LBP
    lbp_imagem = local_binary_pattern(img_segmentada, P=8, R=1, method="default")
    pixels_lbp = lbp_imagem[binario > 0]
    hist_lbp, _ = np.histogram(pixels_lbp, bins=256, range=(0, 256))
    probabilidade_lbp = hist_lbp / hist_lbp.sum()
    media_lbp = np.sum(niveis_lbp * probabilidade_lbp)
    variancia_lbp = np.sum(((niveis_lbp - media_lbp) ** 2) * probabilidade_lbp)
    desvio_lbp = np.sqrt(variancia_lbp)
    suavidade_lbp = 1 - (1 / (1 + variancia_lbp))
    terceiro_momento_lbp = np.sum(((niveis_lbp - media_lbp) ** 3) * probabilidade_lbp)
    bins_ativos = probabilidade_lbp[probabilidade_lbp > 0]
    entropia_lbp = -np.sum(bins_ativos * np.log2(bins_ativos))
    # 7 momentos de Hu
    momentos = cv2.moments(binario)
    hu = cv2.HuMoments(momentos).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-15)

    # Atributos de forma
    rotulado = label(binario)
    propriedades = regionprops(rotulado)
    if len(propriedades) == 0:
        excentricidade = 0
        razão_aspecto = 1
        alongamento = 0
        solidez = 1
        fator_isoperimetrico = 0
        convexidade = 1
    else:
        folha = max(propriedades, key=lambda x: x.area)
        # Exentricidade
        excentricidade = folha.eccentricity
        # Razão de aspecto
        min_row, min_col, max_row, max_col = folha.bbox
        altura = max_row - min_row
        largura = max_col - min_col
        razão_aspecto = altura / largura if largura > 0 else 1
        # Alongamento
        r_circunscrito = folha.equivalent_diameter_area / 2
        mapa_distancia = cv2.distanceTransform(binario, cv2.DIST_L2, 5)
        _, r_inscrito, _, _ = cv2.minMaxLoc(mapa_distancia)
        alongamento = 1 - (r_inscrito / r_circunscrito) if r_circunscrito > 0 else 0
        # Solidez
        solidez = folha.solidity
        # Fator Isoperimetrico
        area = folha.area
        perimetro = folha.perimeter
        fator_isoperimetrico = (
            (4 * np.pi * area) / (perimetro**2) if perimetro > 0 else 0
        )
        # Convexidade
        contornos, _ = cv2.findContours(
            binario, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contornos) > 0:
            maior_contorno = max(contornos, key=cv2.contourArea)
            casco = cv2.convexHull(maior_contorno)
            p_casco = cv2.arcLength(casco, True)
            p_contorno = cv2.arcLength(maior_contorno, True)
            convexidade = p_casco / p_contorno if p_contorno > 0 else 1
        else:
            convexidade = 1
    # Registrar
    registro = {
        "id": nome,
        "y": classe,
        "glcm_dissimilaridade": dissimilaridade,
        "glcm_correlação": correlação,
        "glcm_homogeneidade": homogeneidade,
        "glcm_ASM": asm,
        "glcm_energia": energia,
        "lbp_media": media_lbp,
        "lbp_desviopadrao": desvio_lbp,
        "lbp_suavidade": suavidade_lbp,
        "lbp_Terceiromomento": terceiro_momento_lbp,
        "lbp_entropia": entropia_lbp,
        "forma_excentricidade": excentricidade,
        "forma_razão_aspecto": razão_aspecto,
        "forma_alongamento": alongamento,
        "forma_solidez": solidez,
        "forma_fator_isoperimetrico": fator_isoperimetrico,
        "forma_convexidade": convexidade,
    }
    for i in range(7):
        registro[f"Hu_{i+1}"] = hu_log[i]
    dados_completos.append(registro)
df_features = pd.DataFrame(dados_completos)
df_features.to_csv("Atributos.csv", index=False)
