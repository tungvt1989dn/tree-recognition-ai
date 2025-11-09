# download_images.py
# Sử dụng icrawler để download ảnh theo danh sách tên cây
# Cài đặt: pip install icrawler

from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
import os
import time

# Danh sách cây (có thể chỉnh)
plants = [
 "Kim tiền","Lưỡi hổ","Trầu bà","Sen đá","Vạn niên thanh","Cây dứa cảnh",
 "Đa búp đỏ","Cọ","Hoa hồng","Lan (orchid)","Cẩm tú cầu",
 "Bàng","Ngũ gia bì","Kim ngân","Phát tài","Cau",
 "Trúc","Thường xuân","Nguyệt quế","Phong lộc","Tầm xuân",
 "Sung","Lộc vừng","Mít","Bưởi","Chanh","Ổi",
 "Dừa cảnh","Si","Bạch đàn","Phong","Hồng môn",
 "Xương rồng","Sen","Ficus","Vạn niên (dracaena)",
 "Dương xỉ","Lộc bình (pothos)","Phú quý (aglaonema)","Tùng"
]

ROOT = "plants_dataset"
os.makedirs(ROOT, exist_ok=True)

# số ảnh muốn tải cho mỗi cây
IMAGES_PER_CLASS = 100

for plant in plants:
    safe_name = plant.replace("/", "_").replace("\\","_")
    folder = os.path.join(ROOT, safe_name)
    os.makedirs(folder, exist_ok=True)
    print(f"-> Crawling {plant} into {folder} ...")
    try:
        crawler = GoogleImageCrawler(storage={'root_dir': folder})
        crawler.crawl(keyword=plant, max_num=IMAGES_PER_CLASS, min_size=(200,200))
        # nếu muốn thử Bing thay Google: uncomment:
        # crawler = BingImageCrawler(storage={'root_dir': folder})
        # crawler.crawl(keyword=plant, max_num=IMAGES_PER_CLASS)
    except Exception as e:
        print("Error crawling", plant, e)
    time.sleep(1)  # tránh request liên tục
