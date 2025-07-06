# search_app/views.py
import torch
import numpy as np
import clip
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import time

from django.conf import settings  # 用于获取 MEDIA_URL
from . import global_vars
from .models import Photo

@csrf_exempt
def text_search(request):
    print("【调试】收到请求", flush=True)
    if request.method == "POST":
        try:
            body_unicode = request.body.decode("utf-8")
            data = json.loads(body_unicode)
        except Exception as e:
            print("【调试】解析JSON失败:", e, flush=True)
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        
        cn_text = data.get("query", "")
        print("【调试】请求文本:", cn_text, flush=True)
        if not cn_text:
            return JsonResponse({"error": "No query provided"}, status=400)

        # 翻译阶段
        t0 = time.time()
        try:
            en_text = global_vars.translator(cn_text)[0]["translation_text"]
        except Exception as e:
            print("【调试】翻译失败:", e, flush=True)
            return JsonResponse({"error": "Translation error"}, status=500)
        t1 = time.time()
        print("【调试】翻译完毕, 耗时: {:.2f}秒, 结果: {}".format(t1 - t0, en_text), flush=True)

        # CLIP编码阶段
        t2 = time.time()
        try:
            tokens = clip.tokenize([en_text]).to(global_vars.device)
            with torch.no_grad():
                text_features = global_vars.clip_model.encode_text(tokens)
            text_vector = text_features[0].cpu().numpy().astype("float32").reshape(1, -1)
        except Exception as e:
            print("【调试】CLIP编码失败:", e, flush=True)
            return JsonResponse({"error": "CLIP encoding error"}, status=500)
        t3 = time.time()
        print("【调试】CLIP编码完毕, 耗时: {:.2f}秒".format(t3 - t2), flush=True)

        # FAISS检索阶段（返回 100 张图片）
        t4 = time.time()
        try:
            distances, indices = global_vars.faiss_index.search(text_vector, k=100)
        except Exception as e:
            print("【调试】FAISS检索失败:", e, flush=True)
            return JsonResponse({"error": "FAISS search error"}, status=500)
        t5 = time.time()
        print("【调试】FAISS检索完毕, 耗时: {:.2f}秒".format(t5 - t4), flush=True)
        
        # 映射元数据和构造图片URL
        metadata_df = global_vars.metadata_df  # CSV加载的数据
        results = []
        for rank, idx in enumerate(indices[0]):
            row = metadata_df.iloc[idx]
            photo_id_raw = row.get("photo_id", "").strip()
            # 动态补全：如果不以".jpg"结尾，则补上后缀
            photo_id_display = photo_id_raw
            if not photo_id_display.lower().endswith('.jpg'):
                photo_id_display += '.jpg'
            print("【调试】准备查询数据库中 photo_id =", photo_id_display, flush=True)
            photo_obj = None
            try:
                # 尝试使用补全后的值查询
                photo_obj = Photo.objects.get(photo_id=photo_id_display)
            except Photo.DoesNotExist:
                print("【调试】未找到带后缀的 photo_id:", photo_id_display, flush=True)
                try:
                    # 如果没找到，再用原始值查询
                    photo_obj = Photo.objects.get(photo_id=photo_id_raw)
                    photo_id_display = photo_id_raw  # 更新显示值
                except Photo.DoesNotExist:
                    print("【调试】也未找到原始 photo_id:", photo_id_raw, flush=True)
                    photo_obj = None
            if photo_obj:
                image_url = photo_obj.image.url
                print("【调试】找到对象, image_url =", image_url, flush=True)
            else:
                image_url = ""
            results.append({
                "rank": rank + 1,
                "index": int(idx),
                "distance": float(distances[0][rank]),
                "photo_id": photo_id_display,
                "image_url": image_url
            })
        
        print("【调试】所有步骤完成，返回结果", flush=True)
        return JsonResponse({
            "query": cn_text,
            "translated": en_text,
            "results": results
        }, json_dumps_params={"ensure_ascii": False})
    else:
        return JsonResponse({"error": "Only POST method allowed"}, status=405)