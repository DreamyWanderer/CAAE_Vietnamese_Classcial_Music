# Quy trình xử lý dữ liệu

## 1. Quan sát dữ liệu để nhận dạng đặc tính

- Quan sát file nhạc midi Việt Nam bằng muspy.
- Quan sát file nhạc midi Cổ điển bằng muspy.

## 2. Chuẩn hóa
- Rút lấy phần melody của nhạc midi Việt Nam (manually).
- Xóa toàn bộ khoảng lặng trống ở đầu và cuối bản nhạc (manually).
- Chuyển về resolution 24 (Muspy).
- Bỏ track không tồn tại nốt nhạc (prettyMidi).
- Rút trích nhịp và ô nhịp (Muspy).
- Đoán Major hay Minor (music21).

## 3. Quy trình
- Tải dữ liệu về vào thư mục "Data_original_download".
- Chuyển toàn bộ các file midi vào thư mục "Dataset_original" (Gồm "Vietnamese_dataset" và "Classical_dataset"). Giữ lại track melody của nhạc Việt Nam.
- Tạo các file json chứa thông tin của bản nhạc và chứa nhãn cần thiết (mỗi bản nhạc một file json)
- Dùng các thư viện chuẩn hóa các file dữ liệu này vào thư mục "Dataset_normalized".
- Data augmentation với mỗi bản nhạc vào thư mục "Dataset_normalized".
- Đánh số mỗi bản nhạc và tạo file json chứa các thông tin ứng với từng file trong "Dataset_normalized".
- Tạo các samples vào thư mục "Samples". Tạo một file json cho biết mỗi sample ứng với file .json nào trong "Dataset_normalized".