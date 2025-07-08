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
        
        # 初始化CLIP模型 - 使用 ViT-L/14 (768维特征)
        logger.info("Loading CLIP model...")
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        
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
        image_files = list(self.photos_dir.glob("*.jpg"))
        if not image_files:
            logger.error("No images found in photos directory")
            return
        
        features = []
        image_ids = []
        
        for image_file in tqdm(image_files, desc="Computing features"):
            feature = self.compute_image_features(str(image_file))
            if feature is not None:
                features.append(feature[0])  # Remove batch dimension
                image_ids.append(image_file.stem)
        
        if features:
            self.image_features = np.array(features)
            self.image_ids = image_ids
            
            # 保存特征
            np.save(self.features_file, self.image_features)
            pd.DataFrame(image_ids, columns=['photo_id']).to_csv(self.image_ids_file, index=False)
            
            logger.info(f"Computed and saved features for {len(features)} images")
        else:
            logger.error("No features computed")

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
                
            # 检查特征维度是否正确 (CLIP ViT-L/14 输出维度是768)
            if features.shape[1] != 768:
                logger.warning(f"特征维度不正确：{features.shape[1]}，应该是768")
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
        """生成搜索结果的HTML页面"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_filename = f"search_results_{timestamp}.html"
        html_path = self.results_dir / html_filename
        
        # 复制图片到结果目录
        for result in results:
            image_path = Path(result['image_path'])
            if image_path.exists():
                dest_path = self.images_dir / image_path.name
                shutil.copy2(image_path, dest_path)
                result['local_image_path'] = f"images/{image_path.name}"
        
        # 生成HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>浮世绘搜索结果</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; margin-bottom: 20px; }}
                .result {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; }}
                .result img {{ max-width: 300px; height: auto; }}
                .similarity-score {{ font-weight: bold; color: #007bff; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>浮世绘搜索结果</h1>
                <p><strong>查询:</strong> {query}</p>
                <p><strong>时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        for i, result in enumerate(results):
            html_content += f"""
            <div class="result">
                <h3>结果 {i+1}</h3>
                <p><strong>图片ID:</strong> {result['image_id']}</p>
                <p><strong>相似度得分:</strong> <span class="similarity-score">{result['similarity_score']:.4f}</span></p>
                <img src="{result['local_image_path']}" alt="Image {result['image_id']}">
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # 保存HTML文件
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(html_path)

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("Usage: python search_ukiyoe.py <search_query>")
        sys.exit(1)
    
    query = sys.argv[1]
    searcher = UkiyoeSearcher()
    
    # 检查特征文件
    if not searcher.check_features():
        logger.error("特征文件有问题，请重新计算特征")
        sys.exit(1)
    
    # 执行搜索
    results, html_path = searcher.search(query, k=10)
    
    # 打印结果
    print(f"\n搜索查询: {query}")
    print(f"找到 {len(results)} 个结果:")
    
    for i, result in enumerate(results):
        print(f"{i+1}. {result['image_id']} (相似度: {result['similarity_score']:.4f})")
    
    # 打开结果页面
    print(f"\n结果页面已生成: {html_path}")
    webbrowser.open(f"file://{html_path}")

if __name__ == "__main__":
    main()