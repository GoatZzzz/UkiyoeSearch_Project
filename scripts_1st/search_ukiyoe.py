import sys
import clip
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import webbrowser
import os
import shutil
from datetime import datetime
import pandas as pd
from multilingual_processor import MultilingualProcessor
import logging

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UkiyoeSearcher:
    def __init__(self):
        """初始化搜索器"""
        # 设置基础路径
        self.base_path = Path("/Users/zhu_yangyang/Desktop/UkiyoeSearch_Project")
        
        # 设置设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # 初始化CLIP模型
        logger.info("Loading CLIP model...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # 初始化多语言处理器
        self.multilingual_processor = MultilingualProcessor(device=self.device)
        
        # 设置路径
        self.photos_dir = self.base_path / "ukiyoe_dataset" / "photos"
        self.features_dir = self.base_path / "ukiyoe_dataset" / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # 特征文件路径
        self.features_file = self.features_dir / "features.npy"
        self.image_ids_file = self.features_dir / "photo_ids.csv"
        
        # 设置结果展示相关路径
        self.results_dir = self.base_path / "search_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建图片目录
        self.images_dir = self.results_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载特征
        self.load_features()

    def compute_image_features(self, image_path: str) -> np.ndarray:
        """使用CLIP计算图像特征向量"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model.encode_image(image_input)
                features /= features.norm(dim=-1, keepdim=True)
            
            return features.cpu().numpy()
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None

    def load_features(self):
        """加载或计算图像特征"""
        if self.features_file.exists() and self.image_ids_file.exists():
            logger.info("Loading pre-computed features...")
            self.image_features = np.load(self.features_file)
            self.image_ids = pd.read_csv(self.image_ids_file)['photo_id'].tolist()
            logger.info(f"Loaded features for {len(self.image_ids)} images")
        else:
            logger.info("Computing features for all images...")
            self.compute_and_save_features()

    def compute_and_save_features(self):
        """计算并保存所有图像的特征"""
        features_list = []
        self.image_ids = []
        
        image_files = sorted(list(self.photos_dir.glob("*.jpg")))
        if not image_files:
            raise ValueError(f"No images found in {self.photos_dir}")
        
        for image_file in tqdm(image_files, desc="Computing image features"):
            features = self.compute_image_features(str(image_file))
            if features is not None:
                features_list.append(features)
                self.image_ids.append(image_file.stem)
        
        if not features_list:
            raise ValueError("No features could be computed from the images")
        
        self.image_features = np.vstack(features_list)
        
        # 保存特征和图片ID
        np.save(self.features_file, self.image_features)
        pd.DataFrame({'photo_id': self.image_ids}).to_csv(self.image_ids_file, index=False)
        
        logger.info(f"Computed and saved features for {len(self.image_ids)} images")

    def check_features(self):
        """检查特征文件是否存在且有效"""
        if not self.features_file.exists() or not self.image_ids_file.exists():
            logger.warning("特征文件不存在，需要先计算特征")
            return False
            
        try:
            features = np.load(self.features_file)
            image_ids = pd.read_csv(self.image_ids_file)
            
            if len(features) == 0 or len(image_ids) == 0:
                logger.warning("特征文件是空的，需要重新计算")
                return False
                
            # 检查特征维度是否正确 (CLIP ViT-B/32 输出维度是512)
            if features.shape[1] != 512:
                logger.warning(f"特征维度不正确：{features.shape[1]}，应该是512")
                return False
                
            return True
        except Exception as e:
            logger.error(f"读取特征文件时出错：{e}")
            return False

    def search(self, query: str, k: int = 5):
      """多语言图像搜索"""
      try:
        # 使用多语言处理器处理查询
        with torch.no_grad():
            query_features = self.multilingual_processor(query)
            query_features = query_features.to(self.device)
            
            # 计算相似度
            image_features = torch.tensor(self.image_features).to(self.device)
            similarities = torch.mm(query_features, image_features.T)
            similarities = similarities.cpu().numpy()[0]
            
            # 获取最相似的结果
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            results = []
            for idx in top_indices:
                image_id = self.image_ids[idx]
                image_path = self.photos_dir / f"{image_id}.jpg"
                results.append({
                    'image_id': image_id,
                    'similarity_score': float(similarities[idx]),
                    'image_path': str(image_path)
                })
            
            # 生成结果页面
            html_path = self.generate_results_page(query, results)
            return results, html_path
            
      except Exception as e:
        logger.error(f"Error during search: {e}")
        raise

    def generate_results_page(self, query: str, results: list) -> str:
        """生成HTML结果页面"""
        # 清理旧的图片
        for old_file in self.images_dir.glob("*"):
            try:
                old_file.unlink()
            except:
                pass

        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multilingual Ukiyoe Search Results</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .search-info {{
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .results-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            align-items: start;
        }}
        .result-card {{
            background-color: #fff;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
        }}
        .image-container {{
            width: 100%;
            position: relative;
            margin-bottom: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
            overflow: hidden;
        }}
        .result-card img {{
            width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }}
        .info-container {{
            padding: 10px 0;
        }}
        .similarity-score {{
            color: #2c5282;
            font-weight: bold;
            margin: 5px 0;
        }}
        .image-id {{
            color: #666;
            font-size: 0.9em;
            margin: 5px 0;
        }}
        h1 {{
            color: #2d3748;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="search-info">
        <h1>Multilingual Ukiyoe Search Results</h1>
        <p><strong>Query:</strong> {query}</p>
        <p><strong>Number of results:</strong> {len(results)}</p>
    </div>
    <div class="results-grid">'''

        for result in results:
            image_path = Path(result['image_path'])
            unique_image_name = f"{result['image_id']}_{int(result['similarity_score']*1000):04d}.jpg"
            result_image_path = self.images_dir / unique_image_name
            
            try:
                shutil.copy2(image_path, result_image_path)
                
                # 获取图片尺寸
                with Image.open(image_path) as img:
                    width, height = img.size
                    aspect_ratio = height / width * 100

                html_content += f'''
        <div class="result-card">
            <div class="image-container">
                <img src="images/{unique_image_name}" alt="Image {result['image_id']}">
            </div>
            <div class="info-container">
                <p class="similarity-score">Similarity: {result['similarity_score']:.4f}</p>
                <p class="image-id">Image ID: {result['image_id']}</p>
                <p class="image-id">Dimensions: {width}x{height}</p>
            </div>
        </div>'''
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                continue

        html_content += '''
    </div>
</body>
</html>'''

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"search_results_{timestamp}.html"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(result_file)

def main():
    if not hasattr(sys, 'real_prefix') and not sys.base_prefix != sys.prefix:
        print("Warning: It seems you're not running in a virtual environment!")
    
    print("\nZero-shot Ukiyoe Image Search System")
    print("Enter 'quit' or 'exit' to end the program")
    print("-" * 50)

    try:
        searcher = UkiyoeSearcher()
        
        # 检查特征文件
        if not searcher.check_features():
            print("\n需要先计算图片特征。请运行：")
            print("python scripts/compute_features.py")
            return
        
        while True:
            try:
                query = input("\nEnter your search query ('quit' or 'exit' to exit): ").strip()
                if query.lower() in ['quit', 'exit']:
                    print("\nThank you for using the Ukiyoe Search System!")
                    break
                if not query:
                    continue
                
                try:
                    k = int(input("How many results do you want to see? (default: 5) ") or "5")
                except ValueError:
                    k = 5
                
                results, html_path = searcher.search(query, k)
                
                print(f"\nTop {k} results for query: '{query}'")
                print("-" * 50)
                
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. Image ID: {result['image_id']}")
                    print(f"   Similarity Score: {result['similarity_score']:.4f}")
                
                print(f"\nOpening results in your default web browser...")
                webbrowser.open(f"file://{html_path}")
                
                # 获取用户反馈
                feedback = input("\nWere these results relevant? (y/n/partial): ")
                notes = input("Any specific observations? (Enter to skip): ")
                
                # 保存反馈
                feedback_file = Path(searcher.base_path) / "search_feedback.txt"
                with open(feedback_file, "a", encoding='utf-8') as f:
                    f.write(f"\nTimestamp: {datetime.now()}\n")
                    f.write(f"Query: {query}\n")
                    f.write(f"Results: {[r['image_id'] for r in results]}\n")
                    f.write(f"Feedback: {feedback}\n")
                    if notes:
                        f.write(f"Notes: {notes}\n")
                    f.write("-" * 50 + "\n")
                
            except KeyboardInterrupt:
                print("\nSearch interrupted by user.")
                print("\nThank you for using the Ukiyoe Search System!")
                break
            except Exception as e:
                logger.error(f"Error during search: {e}")
                continue
                
    except Exception as e:
        print(f"Error initializing searcher: {e}")
        return

if __name__ == "__main__":
    main()