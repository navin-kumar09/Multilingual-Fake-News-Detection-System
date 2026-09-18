import pandas as pd
from src.config import DATA_DIR, DATASET_PATHS, binarize_label

def load_fake_true_csv(paths_dict):
    frames = []
    for label_name, path in paths_dict.items():
        if not path.exists(): 
            continue
        df = pd.read_csv(path)
        text_col = "text" if "text" in df.columns else "title"
        if text_col in df.columns:
            sub_df = pd.DataFrame({
                "text": df[text_col], 
                "label": binarize_label(label_name), 
                "source": "ISOT"
            })
            frames.append(sub_df)
    return frames

def load_ifnd(path):
    if not path.exists(): 
        return []
    df = pd.read_csv(path, encoding="latin-1")
    df = df.rename(columns={"Statement": "text", "Label": "raw_label"})
    if "text" in df.columns and "raw_label" in df.columns:
        df["label"] = df["raw_label"].map(binarize_label)
        df["source"] = "IFND"
        return [df[["text", "label", "source"]]]
    return []

def load_liar_tsv(liar_paths):
    frames = []
    for p in liar_paths.values():
        if not p.exists(): 
            continue
        df = pd.read_csv(p, sep="\t", header=None, usecols=[1, 2], names=["raw_label", "text"])
        df["label"] = df["raw_label"].map(binarize_label)
        df["source"] = "LIAR"
        frames.append(df[["text", "label", "source"]])
    return frames

def build_combined_dataset():
    print("📂 Loading and standardizing multi-source datasets...")
    frames = []
    
    frames.extend(load_fake_true_csv(DATASET_PATHS["fake_true_csv"]))
    frames.extend(load_ifnd(DATASET_PATHS["ifnd"]))
    frames.extend(load_liar_tsv(DATASET_PATHS["liar"]))
    
    india_p = DATASET_PATHS["india_csv"]
    if india_p.exists():
        india_df = pd.read_csv(india_p)
        india_df = india_df.rename(columns={
            'Label': 'raw_label', 'label': 'raw_label', 
            'description': 'text', 'headline': 'text', 'Statement': 'text'
        })
        if 'text' in india_df.columns and 'raw_label' in india_df.columns:
            india_df["label"] = india_df["raw_label"].map(binarize_label)
            india_df["source"] = "India_Incidents"
            frames.append(india_df[["text", "label", "source"]])
            
    hindi_folder_base = DATA_DIR / "Hindi_F&R_News"
    hindi_map = {"Hindi_real_news": 1, "Hindi_fake_news": 0}
    for folder_name, lbl in hindi_map.items():
        folder_path = hindi_folder_base / folder_name
        if folder_path.exists():
            txt_data = []
            for txt_file in folder_path.glob("*.txt"):
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                    if content:
                        txt_data.append({"text": content, "label": lbl, "source": "Hindi_FR"})
            if txt_data:
                frames.append(pd.DataFrame(txt_data))
                
    valid_frames = [f for f in frames if not f.empty and 'text' in f.columns]
    if not valid_frames:
        raise ValueError("❌ No valid datasets found! Check file paths.")
        
    df = pd.concat(valid_frames, ignore_index=True)
    df = df[df['label'] != -1]
    df["text"] = df["text"].astype(str).str.strip()
    print(f"✅ Successfully compiled combined dataset: {len(df):,} total rows.")
    return df
