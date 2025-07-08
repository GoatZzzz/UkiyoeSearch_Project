import json
import os
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import logging
import sys
import numpy as np
import pandas as pd
import faiss
import torch
import clip
import re

# 导入全局变量
from . import global_vars

# 设置日志
logger = logging.getLogger(__name__)

# 全局变量存储模型和数据
model = None
index = None
features = None
photo_ids = None

def is_chinese(text):
    """检测文本是否包含中文"""
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    return bool(chinese_pattern.search(text))

def translate_to_english(text):
    """将中文翻译成英文 - 使用mBART模型"""
    if not is_chinese(text):
        return text  # 如果不是中文，直接返回
    
    # 使用全局变量中的mBART翻译器
    if global_vars.translator is None:
        logger.warning("mBART翻译器不可用，返回原文")
        return text
    
    try:
        # 使用mBART翻译器
        result = global_vars.translator(text)
        translated = result[0]['translation_text']
        logger.info(f"mBART翻译: '{text}' -> '{translated}'")
        return translated
    except Exception as e:
        logger.warning(f"mBART翻译失败: {e}")
        return text  # 翻译失败，返回原文

def index_view(request):
    """主页视图"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>浮世绘搜索系统</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .search-box { margin: 20px 0; }
            input[type="text"] { width: 70%; padding: 10px; font-size: 16px; border: 1px solid #ddd; border-radius: 5px; }
            button { padding: 10px 20px; font-size: 16px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            button:disabled { background: #ccc; cursor: not-allowed; }
            .results { margin-top: 20px; }
            .result-item { margin: 10px 0; padding: 10px; border: 1px solid #eee; border-radius: 5px; }
            .status { margin: 10px 0; padding: 10px; background: #e9ecef; border-radius: 5px; }
            .translation-info { font-size: 12px; color: #666; margin-top: 5px; }
            .pagination { margin: 20px 0; text-align: center; }
            .pagination button { margin: 0 5px; min-width: 40px; }
            .pagination .current-page { background: #28a745; }
            .pagination .page-info { margin: 0 10px; font-size: 14px; color: #666; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎨 浮世绘搜索系统</h1>
            <p style="text-align: center; color: #666;">使用ViT-B/32模型，支持中英文搜索，包含58,612张浮世绘图片</p>
            <p style="text-align: center; color: #888; font-size: 14px;">💡 中文查询会自动翻译成英文以提高搜索准确度</p>
            
            <div class="search-box">
                <input type="text" id="queryInput" placeholder="输入搜索关键词，例如：富士山、樱花、武士..." />
                <button onclick="search()">搜索</button>
            </div>
            
            <div id="status" class="status" style="display: none;"></div>
            
            <!-- 分页控件 -->
            <div id="pagination" class="pagination" style="display: none;">
                <button id="prevBtn" onclick="previousPage()">上一页</button>
                <span id="pageInfo" class="page-info"></span>
                <button id="nextBtn" onclick="nextPage()">下一页</button>
                <div style="margin-top: 10px;">
                    跳转到第 <input type="number" id="pageInput" min="1" max="1" style="width: 80px; padding: 5px; text-align: center;" placeholder="页码" /> 页
                    <button onclick="goToPage()" style="margin-left: 5px;">跳转</button>
                </div>
            </div>
            
            <div id="results" class="results"></div>
        </div>

        <script>
            let currentQuery = '';
            let currentPage = 1;
            let totalPages = 1;
            let perPage = 20;
            
            function search(page = 1) {
                const query = document.getElementById('queryInput').value.trim();
                if (!query) {
                    alert('请输入搜索关键词');
                    return;
                }
                
                currentQuery = query;
                currentPage = page;
                
                const statusDiv = document.getElementById('status');
                const resultsDiv = document.getElementById('results');
                const paginationDiv = document.getElementById('pagination');
                
                statusDiv.style.display = 'block';
                statusDiv.innerHTML = '🔍 搜索中...';
                resultsDiv.innerHTML = '';
                paginationDiv.style.display = 'none';
                
                fetch('/search/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: query,
                        page: page,
                        per_page: perPage
                    })
                })
                .then(response => response.json())
                .then(data => {
                    let statusText = `✅ 找到 ${data.total_results} 个结果`;
                    if (data.max_results && data.total_results >= data.max_results) {
                        statusText += ` (显示前${data.max_results}个最相似结果)`;
                    }
                    statusText += ` (模型: ${data.model_type})`;
                    if (data.translated_query && data.translated_query !== data.query) {
                        statusText += `<div class="translation-info">🌐 翻译: "${data.query}" → "${data.translated_query}"</div>`;
                    }
                    statusDiv.innerHTML = statusText;
                    
                    if (data.results && data.results.length > 0) {
                        let html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px;">';
                        data.results.forEach(result => {
                            html += `
                                <div class="result-item" style="text-align: center;">
                                    <img src="${result.image_url}" alt="${result.photo_id}" 
                                         style="max-width: 100%; height: 150px; object-fit: cover; border-radius: 5px;" 
                                         onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZpbGw9IiM5OTkiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5JbWFnZSBOb3QgRm91bmQ8L3RleHQ+PC9zdmc+'" />
                                    <div style="margin-top: 5px; font-size: 12px; color: #666;">
                                        距离: ${result.distance.toFixed(4)}<br/>
                                        排名: #${result.rank}
                                    </div>
                                </div>
                            `;
                        });
                        html += '</div>';
                        resultsDiv.innerHTML = html;
                        
                        // 更新分页信息
                        totalPages = data.total_pages;
                        updatePagination(data);
                    } else {
                        resultsDiv.innerHTML = '<p style="text-align: center; color: #666;">没有找到相关结果</p>';
                        paginationDiv.style.display = 'none';
                    }
                })
                .catch(error => {
                    console.error('搜索错误:', error);
                    statusDiv.innerHTML = '❌ 搜索失败: ' + error.message;
                    paginationDiv.style.display = 'none';
                });
            }
            
            function updatePagination(data) {
                const paginationDiv = document.getElementById('pagination');
                const prevBtn = document.getElementById('prevBtn');
                const nextBtn = document.getElementById('nextBtn');
                const pageInfo = document.getElementById('pageInfo');
                const pageInput = document.getElementById('pageInput');
                
                if (data.total_pages > 1) {
                    paginationDiv.style.display = 'block';
                    
                    // 更新按钮状态
                    prevBtn.disabled = !data.has_previous;
                    nextBtn.disabled = !data.has_next;
                    
                    // 更新页码信息
                    pageInfo.textContent = `第 ${data.current_page} 页 / 共 ${data.total_pages} 页`;
                    
                    // 更新跳转输入框的最大值
                    if (pageInput) {
                        pageInput.max = data.total_pages;
                        pageInput.placeholder = `1-${data.total_pages}`;
                    }
                } else {
                    paginationDiv.style.display = 'none';
                }
            }
            
            function previousPage() {
                if (currentPage > 1) {
                    search(currentPage - 1);
                }
            }
            
            function nextPage() {
                if (currentPage < totalPages) {
                    search(currentPage + 1);
                }
            }

            function goToPage() {
                const pageInput = document.getElementById('pageInput');
                const targetPage = parseInt(pageInput.value, 10);
                if (!isNaN(targetPage) && targetPage >= 1 && targetPage <= totalPages) {
                    search(targetPage);
                    pageInput.value = ''; // 清空输入框
                } else {
                    alert(`请输入1到${totalPages}之间的页码`);
                }
            }
            
            // 回车键搜索
            document.getElementById('queryInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    search();
                }
            });
            
            // 页码输入框回车键跳转
            document.addEventListener('keypress', function(e) {
                if (e.target.id === 'pageInput' && e.key === 'Enter') {
                    goToPage();
                }
            });
        </script>
    </body>
    </html>
    """
    return HttpResponse(html_content)

def load_model_and_index():
    """加载ViT-B/32模型和索引"""
    global model, index, features, photo_ids
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    if model is None:
        model, _ = clip.load("ViT-B/32", device=device)
        logger.info("ViT-B/32 模型加载完成")
    
    # 加载索引
    if index is None:
        index = faiss.read_index("/home/zhu01/UkiyoeSearch_Project/ukiyoe_dataset/index/faiss_index_ivf_vit_l14.index")
        logger.info("FAISS索引加载完成")
    
    # 加载特征和图片ID
    if features is None:
        features = np.load("/home/zhu01/UkiyoeSearch_Project/ukiyoe_dataset/features/features.npy")
        logger.info("特征文件加载完成")
    
    if photo_ids is None:
        photo_ids = pd.read_csv("/home/zhu01/UkiyoeSearch_Project/ukiyoe_dataset/features/photo_ids.csv")
        logger.info("图片ID文件加载完成")

@csrf_exempt
@require_http_methods(["POST"])
def search_api(request):
    """搜索API - 使用ViT-B/32，支持中文翻译和分页"""
    try:
        # 加载模型和索引
        load_model_and_index()
        
        # 解析请求
        data = json.loads(request.body)
        query = data.get('query', '')
        page = int(data.get('page', 1))  # 页码，默认为1
        per_page = int(data.get('per_page', 20))  # 每页数量，默认为20
        
        # 设置最大搜索结果数，支持更多页面浏览
        # 10000个结果可以支持500页（每页20个）的浏览
        MAX_SEARCH_RESULTS = 10000
        
        logger.info(f"搜索查询: {query}, 页码: {page}, 每页: {per_page}")
        
        if not query:
            return JsonResponse({'error': '查询不能为空'}, status=400)
        
        # 如果是中文，翻译成英文
        original_query = query
        translated_query = translate_to_english(query)
        
        # 编码查询文本（使用翻译后的文本）
        device = next(model.parameters()).device
        tokens = clip.tokenize([translated_query]).to(device)
        
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 一次性搜索大量结果以支持多页浏览
        query_vector = text_features.cpu().numpy().astype('float32')
        distances, indices = index.search(query_vector, MAX_SEARCH_RESULTS)
        
        # 构建所有搜索结果
        all_results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            photo_id = photo_ids.iloc[idx]['photo_id']
            
            # 构建图片URL
            if photo_id.endswith('.jpg'):
                image_url = f"/media/{photo_id}"
            else:
                image_url = f"/media/{photo_id}.jpg"
            
            all_results.append({
                'id': int(idx),
                'photo_id': photo_id,
                'image_url': image_url,
                'distance': float(distance),
                'rank': i + 1
            })
        
        # 计算总页数（基于所有搜索结果）
        total_results = len(all_results)
        total_pages = (total_results + per_page - 1) // per_page
        
        # 验证页码有效性
        if page > total_pages:
            page = total_pages
        if page < 1:
            page = 1
        
        # 从所有结果中提取当前页的结果
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_results = all_results[start_idx:end_idx]
        
        # 返回结果，包括翻译信息和分页信息
        response_data = {
            'results': page_results,
            'query': original_query,
            'model_type': 'vit_b32',
            'total_results': total_results,
            'max_results': MAX_SEARCH_RESULTS,
            'current_page': page,
            'per_page': per_page,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_previous': page > 1,
            'search_time': 'N/A'
        }
        
        # 如果进行了翻译，添加翻译信息
        if translated_query != original_query:
            response_data['translated_query'] = translated_query
        
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"搜索错误: {e}")
        return JsonResponse({'error': str(e)}, status=500)