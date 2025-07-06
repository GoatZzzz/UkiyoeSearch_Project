from django.db import models

class Photo(models.Model):
    photo_id = models.CharField(max_length=255, unique=True, help_text="图片命名")
    # 增加 max_length 参数，例如设置为 500
    image = models.ImageField(upload_to='photos/', max_length=500, help_text="上传的图片文件")

    def __str__(self):
        return self.photo_id