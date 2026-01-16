import collections.abc as abc


# 字典常用方法
def collec_dict():
    d = {}
    print(f"d isinstance acb.Mapping {isinstance(d, abc.Mapping)}")
    # 支持列表推导
    tu_list = [(1, "ONE"), (2, "TWO"), (3, "THREE")]
    d1 = {k: v for v, k in tu_list}
    print(f"d1={d1}")
    # 常用方法-查找
    print(
        f"d1 contains ONE {d1.__contains__('ONE')}",
        f"d1 get ONE {d1.__getitem__('ONE')}",
        f"d1 get ONE {d1.__getattribute__('__getitem__')('ONE')}",
        f"d1 get ONE {d1.get('ONE')}",
        f"d1 get ONE {d1['ONE']}",
    )
    # 常用方法-增加/更新
    d2 = d1 | {"FOUR": 4, "ONE": 11}
    print(
        f"d1 不变 {d1}",
        f"d2 add/update {d2}",
    )

    d1["ONE"] = 11
    print(d1)

    # 常用方法-删除
    if "ONE" in d1:
        del d1["ONE"]
    print(d1)

    # 遍历
    for k in d1.keys():
        print(f"k-v {k + "-" + str(d1[k])}")
    for k, v in d1.items():
        print(f"k-v {k + "-" + str(v)}")

    # 字典解包
    u = {"name": "felix", "age": 18}
    print("姓名={name}，age={age}".format(**u))
    print("姓名={name}，age={age}".format(name="felix", age=18))


# 集合常用工具
def collec_list():
    l = ["spam", "spam", "eggs", "spam"]
    # 去重
    l1 = list(set(l))
    print(l1)
    a = {"A", "B", "C", "D"}
    b = {"C", "D", "E", "F"}
    # 集合运算
    print(f"合集 {a | b}", f"交集 {a & b}", f"ab差集 {a - b}", f"ba差集{b - a}")


# 切片 list[start:stop:step]  start stop 指的是索引 从0开始
# 左闭右开，step 控制方向和跨度
def collec_slice():
    nums = list(range(10))
    # 0,1,2,3,4,5,6,7,8,9
    print(f"原始列表 nums={nums}")

    # 基础切片
    print("索引2到4的元素 =", nums[2:5])
    print("索引小于5的元素 =", nums[:5])
    print("索引大于等于5的元素 =", nums[5:])
    # 反向切片
    print("反转 =", nums[::-1])
    print("最后三个 =", nums[-3:])
    print("去掉最后三个 =", nums[:-3])

    s = "hello python"
    print("s[0:5] = ", s[0:5])
    print("s[：：-1] = ", s[::-1])


def collec_comprehension():
    nums = list(range(1, 11))
    print(f"原始 nums={nums}")
    # 基础变换
    print("map平方 = ", [x**2 for x in nums])
    # 带条件过滤
    print("filter偶数 = ", [x for x in nums if x % 2 == 0])
    # 条件表达式（三元）
    labels = ["even" if x % 2 == 0 else "odd" for x in nums]
    print("奇偶 labels = ", labels)
    # 多层循环
    pairs = [(x, y) for x in [1, 2, 3] for y in ["A", "B"]]
    print("多层循环 pairs =", pairs)
    # 字符串处理
    words = [" python ", " java", "go ", " rust "]
    clean_words = [w.strip().upper() for w in words]
    print("字符串处理 clean_words =", clean_words)
    # 和 set / dict 配合
    unique_lengths = {len(w) for w in words}
    print("set 推导 unique_lengths =", unique_lengths)
    word_len_map = {w.strip(): len(w.strip()) for w in words}
    print("dict 推导 word_len_map =", word_len_map)


if __name__ == "__main__":
    collec_comprehension()
