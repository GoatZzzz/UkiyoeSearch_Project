# search_app/management/commands/import_photos.py
import csv
import os
from django.core.management.base import BaseCommand
from django.core.files import File
from search_app.models import Photo
from django.conf import settings

class Command(BaseCommand):
    help = '从 CSV 导入照片数据，并上传图片文件（方案 A）'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='CSV 文件的完整路径')
        parser.add_argument(
            '--source',
            type=str,
            default='/Users/zhu_yangyang/Desktop/UkiyoeSearch_Project/ukiyoe_dataset/photos',
            help='图片文件的原始存放目录'
        )

    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']
        source_dir = kwargs['source']
        self.stdout.write(f"开始导入 CSV 数据：{csv_file}")
        count = 0

        with open(csv_file, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                photo_id_raw = row.get('photo_id', '').strip()
                if not photo_id_raw:
                    continue

                # 检查 photo_id_raw 是否以 .jpg 结尾，如没有，则补上
                if not photo_id_raw.lower().endswith('.jpg'):
                    actual_filename = photo_id_raw + '.jpg'
                else:
                    actual_filename = photo_id_raw

                self.stdout.write(f"导入记录：原始 photo_id = {photo_id_raw}, 实际文件名 = {actual_filename}")

                # 构造图片文件的完整路径
                source_file_path = os.path.join(source_dir, actual_filename)
                if not os.path.exists(source_file_path):
                    self.stdout.write(self.style.WARNING(f"文件不存在: {source_file_path}"))
                    continue

                # 使用 get_or_create 防止重复导入
                obj, created = Photo.objects.get_or_create(photo_id=photo_id_raw)
                if created:
                    with open(source_file_path, 'rb') as image_file:
                        obj.image.save(actual_filename, File(image_file), save=True)
                    count += 1
        self.stdout.write(self.style.SUCCESS(f'成功导入 {count} 条照片数据'))