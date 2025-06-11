import pandas as pd

# 파일 로드
ppi_df = pd.read_csv("STRING_ppi_edgelist.tsv", sep="\t")
gene_df = pd.read_csv("merged_result.csv")

# GeneID 매핑 딕셔너리 생성
gene_dict = dict(zip(gene_df["gene"], gene_df["GeneID"]))

# partner1과 partner2를 GeneID로 변환
ppi_df["partner1"] = ppi_df["partner1"].map(gene_dict)
ppi_df["partner2"] = ppi_df["partner2"].map(gene_dict)

# NaN 값 제거 (매핑되지 않은 값 제외)
ppi_df = ppi_df.dropna(subset=["partner1", "partner2"]).astype(int)

# confidence 컬럼 제외하고 저장
ppi_df[["partner1", "partner2"]].to_csv("STRING_ppi_edgelist_mapped.tsv", sep="\t", index=False, header=False)
