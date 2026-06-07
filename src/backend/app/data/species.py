from csv import DictReader
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


FALLBACK_SPECIES = [
    {
        "id": "sp001",
        "common_name": "乌鸫",
        "scientific_name": "Turdus merula",
        "family": "鸫科",
        "description": "城市、公园和林缘常见鸟类，鸣声圆润清亮，适合做演示样例。",
        "habitat": "林地、城市绿地、农田边缘",
    },
    {
        "id": "sp002",
        "common_name": "麻雀",
        "scientific_name": "Passer montanus",
        "family": "雀科",
        "description": "常见小型鸟类，叫声短促重复，在环境噪声中也较容易听到。",
        "habitat": "居民区、农田、校园",
    },
    {
        "id": "sp003",
        "common_name": "白头鹎",
        "scientific_name": "Pycnonotus sinensis",
        "family": "鹎科",
        "description": "南方城市常见鸟，叫声变化较多，适合后续展示混淆分析。",
        "habitat": "灌丛、果园、城市绿地",
    },
    {
        "id": "sp004",
        "common_name": "喜鹊",
        "scientific_name": "Pica pica",
        "family": "鸦科",
        "description": "体型较大的鸦科鸟类，叫声粗哑，和小型鸣禽差异明显。",
        "habitat": "村镇、农田、防护林",
    },
    {
        "id": "sp005",
        "common_name": "家燕",
        "scientific_name": "Hirundo rustica",
        "family": "燕科",
        "description": "飞行鸣叫频繁，声音轻快，后续可结合时间序列模型演示。",
        "habitat": "村落、河流附近、开阔地",
    },
    {
        "id": "sp006",
        "common_name": "大山雀",
        "scientific_name": "Parus major",
        "family": "山雀科",
        "description": "鸣叫节奏感明显，常见于林地和城市公园。",
        "habitat": "阔叶林、混交林、公园",
    },
    {
        "id": "sp007",
        "common_name": "珠颈斑鸠",
        "scientific_name": "Spilopelia chinensis",
        "family": "鸠鸽科",
        "description": "叫声低沉重复，频谱形态和高频鸣禽差别较大。",
        "habitat": "城市绿地、农田、村镇",
    },
    {
        "id": "sp008",
        "common_name": "东方大苇莺",
        "scientific_name": "Acrocephalus orientalis",
        "family": "苇莺科",
        "description": "湿地常见鸣禽，鸣声复杂，适合作为难分类样例。",
        "habitat": "芦苇荡、湿地、湖泊边缘",
    },
]


def _load_taxonomy_species():
    taxonomy_path = ROOT / "data" / "taxonomy.csv"
    if not taxonomy_path.exists():
        return []

    rows = []
    with taxonomy_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in DictReader(handle):
            species_id = str(row.get("primary_label", "")).strip()
            if not species_id:
                continue
            common_name = str(row.get("common_name", "")).strip() or species_id
            scientific_name = str(row.get("scientific_name", "")).strip() or common_name
            class_name = str(row.get("class_name", "")).strip() or "Unknown"
            rows.append(
                {
                    "id": species_id,
                    "common_name": common_name,
                    "scientific_name": scientific_name,
                    "family": class_name,
                    "description": f"BirdCLEF2026 taxonomy class {species_id} ({common_name}).",
                    "habitat": "Pantanal, Brazil / BirdCLEF2026 recording region",
                }
            )
    return rows


SPECIES = _load_taxonomy_species() or FALLBACK_SPECIES
SPECIES_BY_ID = {item["id"]: item for item in SPECIES}


def list_species():
    return SPECIES


def get_species(species_id):
    return SPECIES_BY_ID.get(str(species_id))


def species_label(label):
    label = str(label)
    species = get_species(label)
    if species:
        return species
    return {
        "id": label,
        "common_name": label,
        "scientific_name": label,
        "family": "Unknown",
        "description": f"Model label {label}.",
        "habitat": "Unknown",
    }
