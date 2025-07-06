import matplotlib.pyplot as plt
import cv2

def show_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # cv2 读进来是 BGR，需要转换到 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

# 假如 metadata_df 里有个列 'image_path' 是文件路径
row0 = metadata_df.iloc[0]
print("photo_id =", row0['photo_id'])
show_image(row0['image_path'])  # 显示原图

row1 = metadata_df.iloc[9390]
print("photo_id =", row1['photo_id'])
show_image(row1['image_path'])
# 依次查看 ...