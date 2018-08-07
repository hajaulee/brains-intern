import cv2  # Import OpenCv library
import numpy as np
import sys
import os

total_trained_data = []
admissible_data_area = []
test_folder_path = "./dataset_01/test"
train_folder_path = "./dataset_01/train/"


def between(a, b, c):  # Return true if a between b and c
    if b < c:
        return b < a < c
    else:
        return c < a < b


def find_nearest(vmin, vmax, value, get_max=False, get_min=False):
    if get_max:  # If priority for max value return max
        return vmax
    if get_min:  # If priority for min value return min
        return vmin
    if vmax < value:  # If value is greater than max return max
        return vmax
    if vmin > value:  # If value is smaller than min return min
        return vmin
    return value  # if value between min and max return value


def dark_area(array):  # Get dark area in image (white background)
    white = np.array([255, 255, 255])
    start = 0
    end = 0
    for i in range(len(array)):
        if start == 0 and (array[i] != white).any():
            start = i
        if start != 0 and (array[i] == white).all():
            end = i - 1
        if start != 0 and end != 0:
            return start, end
    return start, end


def learn(funct, array, start=0):
    a = []
    b = []
    avg = []
    dis = array[0] - start
    if dis < 5:
        for x in range(start, array[0] + 1):
            a.append([x ** 5, x ** 4, x ** 3, x ** 2, x, 1])
            b.append(funct[x])
        a = np.array(a)
        if dis < 1:
            return
        while dis + 1 < len(a[0]):
            a = np.delete(a, 0, 1)
        avg = np.linalg.solve(a, b)
        for i in (-1 - dis, -1):
            array[i - 1] = avg[i]
    else:
        for x in range(start, array[0] + 1):
            # Thêm phương trình và giá trị hàm số tại điểm x
            a.append([x ** 5, x ** 4, x ** 3, x ** 2, x, 1])
            b.append(funct[x])
            if x >= start + 5:
                # Giải hệ phương trình để tìm tham số cho đồ thị bậc 5
                avg.append(np.linalg.solve(a, b))
                del a[0]  # Xóa phương trình của điểm cũ
                del b[0]  # Xóa Giá trị tại điểm cũ, để luôn có hệ phương trình 6 ẩn
        avg = np.array(avg)
        for i in range(5):
            array[i + 1] = sum(avg[:, i]) / len(avg[:, i])


def load_image(iname, debug=False):
    im = cv2.imread(iname)
    funct = [0]  # Lưu chuỗi các giá trị của hàm số
    for x in range(len(im[0])):
        start, end = dark_area(im[:, x])
        if start != 0 and end != 0:
            if len(funct) > 2 and funct[-2] > funct[-3] + 20:  # Ưu tiên lấy giá trị max khi hàm số tăng nhanh
                funct.append(find_nearest(start, end, funct[-1], get_max=True))
            elif len(funct) > 2 and funct[-2] < funct[-3] - 20:  # Ưu tiên lấy giá trị min khi hàm số giảm nhanh
                funct.append(find_nearest(start, end, funct[-1], get_min=True))
            else:
                funct.append(find_nearest(start, end, funct[-1]))  # Lấy giá trị gần bằng điểm cận kề
    del funct[0]
    if debug:
        print(funct)
        print("____________________________________")
    length = len(funct)
    #   Mỗi phần tử của graph_data là đại diện cho 1 đoạn(6 đoạn)
    #   Trong 1 phần tử số đầu là vị trí kết thức đoạn,
    #   các số sau là tham số hàm bậc 5 của đoạn đó.
    #   Số cuối lưu cực trị của đoạn
    graph_data = [[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [length - 1, 0, 0, 0, 0, 0, 0, 0]]

    # Tính giới hạn và học đoạn 1 (Đoạn bằng)
    for x in range(length):
        if funct[x] < funct[0] - 2:
            graph_data[0][0] = x
            break
    graph_data[0][7] = funct[graph_data[0][0]]
    learn(funct, graph_data[0])
    if debug:
        print(0, graph_data[0])

    # Tính giới hạn và học đoạn 2 (Đoạn lên nhanh)
    for x in range(graph_data[0][0], length):
        if funct[x] < funct[x - 1] and funct[x] <= funct[x + 1]:
            graph_data[1][0] = x
            break
    graph_data[1][7] = funct[graph_data[1][0]]
    learn(funct, graph_data[1], graph_data[0][0])
    if debug:
        print(1, graph_data[1])

    # Tính giới hạn và học đoạn 3 (Đoạn xuống nhanh)
    for x in range(graph_data[1][0], length):
        if funct[x - 1] < funct[x] <= funct[x + 1]:
            graph_data[2][0] = x
            break
    graph_data[2][7] = funct[graph_data[2][0]]
    learn(funct, graph_data[2], graph_data[1][0])
    if debug:
        print(2, graph_data[2])

    # Tính giới hạn và học đoạn 4 (Đoạn lên quay lại)
    for x in range(graph_data[2][0], length):
        if funct[x] < funct[x - 1] and funct[x] <= funct[x + 1]:
            graph_data[3][0] = x
            break
    graph_data[3][7] = funct[graph_data[3][0]]
    learn(funct, graph_data[3], graph_data[2][0])
    if debug:
        print(3, graph_data[3])

    # Tính giới hạn và học đoạn 5 (Đoạn xuống dốc)
    for x in range(graph_data[3][0], length):
        if funct[x] > funct[x - 1] and funct[x] == funct[x + 1]:
            graph_data[4][0] = x
            break
    graph_data[4][7] = funct[graph_data[4][0]]
    learn(funct, graph_data[4], graph_data[3][0])
    if debug:
        print(4, graph_data[4])

    # Tính giới hạn và học đoạn 6 (Đoạn kết thức)
    graph_data[5][7] = funct[graph_data[5][0]]
    learn(funct, graph_data[5], graph_data[4][0])
    if debug:
        print(5, graph_data[5])
    return graph_data


def train():
    global train_folder_path
    if train_folder_path[-1] != "/":
        train_folder_path += "/"
    train_folder = os.listdir(train_folder_path)
    i = 0
    # Duyệt toàn bộ file trong thư mục train
    for file in train_folder:
        if ".png" in file:
            # Lấy đường dẫn file
            file = train_folder_path + file
            # Phân tích và đưa dữ liệu ảnh vào mảng dữ liệu đã huấn luyện
            total_trained_data.append(load_image(file))
            print(">>>>>>   Training successfully:", i, file)
            i += 1
    # L duyệt qua 6 đoạn
    for L in range(len(total_trained_data[0])):
        D = []  # Các cặp min max của các tham số từng đoạn
        # k duyệt qua 8 tham số của 1 đoạn
        for k in range(len(total_trained_data[0][0])):
            # Lấy cột L,k tức cột tham số thứ k, của đoạn L, trong tất cả các ảnh của tập huấn luyện
            col = np.array(total_trained_data)[:, L, k]
            minc = np.min(col)
            maxc = np.max(col)
            D.append([minc, maxc])
        admissible_data_area.append(D)
    try:
        trained_data_file = open("trained_data", "w")
        print(admissible_data_area, file=trained_data_file)
        trained_data_file.close()
    except IOError:
        print("Can not create trained data file!")
        exit(-1)
    else:
        print("Create trained data file successful!")


def test():
    global test_folder_path
    global admissible_data_area
    if test_folder_path[-1] != "/":
        test_folder_path += "/"
    try:
        trained_data_file = open("trained_data", "r")
        admissible_data_area = eval(trained_data_file.readline())
        trained_data_file.close()
    except IOError:
        print("Can not read trained data file!")
        print("Type 'python main.py train' to train first!")
        show_usage()
        exit(-1)
    else:
        print("Read trained data file successful!")
    test_folder = os.listdir(test_folder_path)
    index = 0  # Số thứ tự ảnh (Bằng số ảnh khi duyệt qua tất cả ảnh)
    correct = 0  # Đếm số ảnh nhận dạng đúng
    for file in test_folder:
        if ".png" in file:
            f = file
            file = test_folder_path + file
            # print("processing:", file)
            a = load_image(file)
            # print(a)
            admissible = True  # Kiểm tra ảnh có phải normal không
            for i in range(len(a)):  # Duyệt qua các đoạn của biểu đồ (6 đoạn)
                # Duyệt các tham số của 1 đoạn
                # Số 1 (giới hạn của đoạn)
                # Số 2-7 (Tham số hàm bậc 5 của đoạn)
                # Số cuối Giá trị tại giới hạn của đoạn
                for j in range(len(a[0])):
                    v_min = admissible_data_area[i][j][0]
                    v_max = admissible_data_area[i][j][1]
                    if a[i][j] < v_min or a[i][j] > v_max:  # Nếu giá trị vượt qua vùng cho phép đánh dấu anomaly

                        # Cho phép sai số nhỏ tại đoạn 4 (Đoạn lên dốc)
                        if i == 3 and 1.05 * v_max > a[i][j] > 1.05 * v_min:
                            # print(a[i][j], 1.05*v_min, 1.05*v_max)
                            continue
                        if i == 0 and j in range(2, 6):
                            if between(a[i][j], 1.9 * v_max, 2.0 * v_max) or between(a[i][j], 1.9 * v_min, 2.0 * v_min):
                                continue
                            # else:
                            #     print(2.0 * v_max , a[i][j] , 1.9 * v_max)
                            #     print(2.0 * v_min , a[i][j] , 1.9 * v_min)
                        admissible = False
                        # print(i,j)
                        # print(a[i][j], v_min, v_max)

            if admissible:  # Nếu ảnh được đánh dấu normal
                print(">>> normal:  %-5d%-20s" % (index, f), end="")
                # Xét tên ảnh nếu không chứa từ anomaly thì nhận dạng đúng
                if "anomaly" not in f:
                    print("Correct recognition")
                    correct += 1
                else:
                    print("Wrong recognition")
            else:  # Nếu ảnh được đánh dấu anomaly
                print(">>> anomaly: %-5d%-20s" % (index, f), end="")
                # Xét tên ảnh nếu chứa từ anomaly thì nhận dạng đúng
                if "anomaly" in f:
                    print("Correct recognition")
                    correct += 1
                else:
                    print("Wrong recognition")
            index += 1
    print("Pass: ", correct, "/", index, "test")


def show_usage():
    print("\nusage: python main.py [train/test] [-p test_folder_path/train_folder_path]")
    print("For example:\t python main.py")
    print("\t\t python main.py train")
    print("\t\t python main.py test ./data_set_01/test")


if __name__ == "__main__":
    try:

        if "train" in sys.argv:
            if "-p" in sys.argv:
                train_folder_path = sys.argv[sys.argv.index("-p") + 1]
            train()
        elif "test" in sys.argv or len(sys.argv) == 1:
            if "-p" in sys.argv:
                test_folder_path = sys.argv[sys.argv.index("-p") + 1]
            test()
        else:
            show_usage()
    except IndexError:
        show_usage()
