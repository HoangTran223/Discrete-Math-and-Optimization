# MODULE:
import math, random
import math as Toan                         # Thay đổi tên 
from math import sqrt, pi                   # Chỉ thêm 1 vài phần của module
from math import sqrt as canbac             # Thay đổi tên của sqrt

# MODULE RANDOM:
import random

print(random.random()*96.4)                 # random()*x trả về 1 số thực ngẫu nhiên thuộc (0,x)

print(random.randint(3,14))                 # randint() trả về 1 số nguyên thuộc [3,14], cận phải nguyên

print(random.randrange(2,13,3))             # trả về 1 số nguyên thuộc [2,13), nhảy bước 3, cận nguyên

# Seed(): 
random.seed(15)
print( random.random())
print( random.random())                     # Giá trị ngẫu nhiên sau mỗi lần
# Tức cái seed này nó lưu lại chuỗi random
# Khi cần ta có thể biết cxac chuỗi số giả ngẫu nhiên, nên rất hữu ích cho việc chạy thử nghiệm, ML,...
random.seed(13)
print( random.random())
print( random.random())

mylist = list(map(float, input().split()))
print(random.choice(mylist))                # Trả về 1 ptu dc chọn ngẫu nhiên từ 1 tập hợp, ds, chuỗi kí tự,...

random.shuffle(mylist)                      # Sắp xếp ngẫu nhiên các ptu trong tập hợp
print(mylist)

# BÀI TẬP: tạo ngẫu nhiên mật khẩu có ký tự hoa, chữ số, ký tự ddbiet, độ dài 8
import random
DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 

LOCASE_CHARACTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                     'i', 'j', 'k', 'm', 'n', 'o', 'p', 'q',
                     'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
                     'z']

UPCASE_CHARACTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                     'I', 'J', 'K', 'M', 'N', 'O', 'p', 'Q',
                     'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
                     'Z']

SYMBOLS = ['@', '#', '$', '%', ':', '?', '.', '/', '|', '~', '>',
           '*', '(', ')', '<']

rand_digit = random.choice(DIGITS)
rand_upper = random.choice(UPCASE_CHARACTERS)
rand_lower = random.choice(LOCASE_CHARACTERS)
rand_symbol = random.choice(SYMBOLS)
temp_pass = [rand_digit, rand_upper, rand_lower, rand_symbol] 

MAX_LEN = 12
COMBINED_LIST = DIGITS + UPCASE_CHARACTERS + LOCASE_CHARACTERS + SYMBOLS
for x in range(MAX_LEN - 4):
    temp_pass.append(random.choice(COMBINED_LIST))

random.shuffle(temp_pass)
password = ""
for x in temp_pass:
        password = password + x

print("Password generated: " + password)


# Sử dụng module tự định nghĩa:
import test_module
mylist = ["Hoàng", "Hải", "Hùng", "Hòa"]
mylist1 = test_module.mylist2
MYLIST = mylist + mylist1
for i in MYLIST:
    test_module.my_function(i)          # Nó sẽ thực hiện hết các code của file test_module

# Module chỉ dc tải về duy nhất 1 lần
# Muốn thay đổi yếu tố trong module cũ, mà cái mới dc sdung nó, ta dùng hàm reload():
from importlib import reload
reload(test_module)

