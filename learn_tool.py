import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum, unique
from pydantic import BaseModel


def tool_uuid():
    user_id = uuid.uuid4()
    print(f"生成的ID:{user_id}")
    # 去除-
    clean_id = user_id.hex
    print(f"生成的ID:{clean_id}")
    # 长度
    print(len(clean_id))
    x_id = user_id.hex[:16]
    print(f"生成的ID:{x_id}")
    # 长度
    print(len(x_id))


def tool_time():
    # 获取当前时间
    now = datetime.now()
    # 日期转字符串
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    # 字符串转日期
    now_obj = datetime.strptime(now_str, "%Y-%m-%d %H:%M:%S")
    # 日期转秒级时间戳
    ts_second = int(now.timestamp())
    # 日期转毫秒级时间戳
    ts_millsecond = int(now.timestamp() * 1000)
    print("now_obj", now_obj)
    print("ts_second", ts_second)
    print("ts_millsecond", ts_millsecond)
    # 秒级时间戳转日期
    dt_utc_obj = datetime.fromtimestamp(ts_second, tz=timezone.utc)
    print("dt_utc_obj", dt_utc_obj)
    # 秒级时间戳转日期
    dt_utc8_obj = datetime.fromtimestamp(ts_second, tz=timezone(timedelta(hours=8)))
    print("dt_utc8_obj", dt_utc8_obj)
    # 当前时间 +2 hours
    print(now + timedelta(days=1, hours=2))
    # 获取当前是周几 周一到周日对应1到7
    print(now.isoweekday())


@unique
class Weekday(Enum):
    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6
    SUNDAY = 7


def tool_enum():
    days = [d for d in Weekday]
    print(days)
    try:
        print(Weekday["MONDAY"])
    except Exception as e:
        print(f"发生未知异常: {repr(e)}")

    try:
        print(Weekday(1))
    except Exception as e:
        print(f"发生未知异常: {repr(e)}")


# 列举所有内置函数（不需要import 直接使用）
def tool_buildinfunctions():
    import builtins

    print(dir(builtins))


# 第一梯队：基础交互与类型转换（使用频率最高）
def tool_infunc_1():
    # 将信息输出到控制台
    print("hello")
    # 返回对象（字符串、列表、字典等）的长度或元素个数
    print(len("Hello"), len([1, 2, 3]), len({"A": 1, "B": 2}))
    # 将对象转换为字符串类型
    print(str(Weekday(1)))
    # 将数字或字符串转换为整数。
    print(int(datetime.now().timestamp()))
    # 返回对象的类型，常用于调试检查数据。
    print(type(Weekday(1)))
    # 将可迭代对象转换为列表
    print(list(Weekday))
    # 创建或将键值对转换为字典
    print(dict(name="张三", age=25, city="北京"))


# 第二梯队：循环、范围与序列处理
def tool_infunc_2():
    # 生成一个不可变的数字序列，常用于 for 循环
    print([x for x in range(1, 10)])
    # 在遍历序列时，同时返回索引和元素（非常实用）
    l = [(i, v) for i, v in enumerate(Weekday)]
    print(l)
    # 将多个可迭代对象包装成一个个元组，用于并行遍历。
    print([v[1] for v in zip(["A", "B", "C"], [1, 2, 3])])
    # 将多个可迭代对象包装成一个个元组，用于并行遍历。
    print(sorted([3, 1, 2]))
    # 反转序列
    print([x for x in reversed(Weekday)])


# 第三梯队：数值运算与逻辑判断
def tool_infunc_3():
    arr = [1, 2, 3]
    # 对序列中的数值进行求和。
    print(sum(arr), min(arr), max(arr))
    # 绝对值
    print(abs(-1))
    # 检查序列中是否存在（或全部为）真值。
    print(any(i > 2 for i in arr))
    print(all(i > 2 for i in arr))


# 第四梯队：对象与调试工具
def tool_infunc_4():
    x = Weekday["MONDAY"]
    print(isinstance(x, Weekday))
    print(isinstance(1, int))
    d: float = 3.1415
    print(isinstance(d, float))
    print(isinstance(d, str))
    print(isinstance(d, bool))
    print(isinstance(d, int))
    print(isinstance(d, list))
    print(isinstance((), tuple))
    print(f"x是否为null {x is None}")
    print(f"x.name = {getattr(x, 'name')}", f"x.value = {getattr(x, 'value')}")


# 序列处理工具 (Tool List/Sequence)
def tool_sequence():
    arr = [x for x in range(1, 7)] + [3, 4, 5]
    print(arr)
    unique_arr = list(set(arr))
    print(unique_arr)
    # 列表切片 (Slicing) - [start:stop:step]
    print(f"前三个: {arr[:3]}")
    print(f"二到四个: {arr[1:4]}")
    print(f"后两个: {arr[-2:]}")
    print(f"翻转列表: {arr[::-1]}")
    # 列表推导式 (Filtering & Mapping)
    e = [x**2 for x in arr if x % 2 == 0]
    print(f"偶数的平方: {e}")


# 字符串高级处理 (Tool String)
def tool_string():
    str = "  Python-Tools-2026\n  "
    str_s = str.strip()
    print(f"去除两端的空格换行: {str_s}")
    # 拆分合并
    arr = str_s.split("-")
    print(f"-拆分: {arr}")
    print(f"_合并: {'_'.join(arr)}")
    # 判断开头结尾
    print(
        f"是否以Py开头: {str_s.startswith('Py')}",
        f"是否以2026结尾: {str_s.endswith('2026')}",
    )
    # 字符串0填充
    print(f"{'42'.zfill(5)}")


class Name:

    def __init__(self, name):
        self.name = name


class User:

    def __init__(self, id, person):
        self.id = id
        self.person = Name(**person) if isinstance(person, dict) else person


# JSON 与 字典处理 (Tool JSON/Dict)
def tool_json():
    import json

    data = {"id": 1, "person": {"name": "er"}}
    arr = [{"id": 1, "person": {"name": "er"}}, {"id": 2, "person": {"name": "san"}}]

    data_str = json.dumps(data)
    arr_str = json.dumps(arr)

    data_obj = json.loads(data_str)
    arr_obj = json.loads(arr_str)
    print(f"data_obj type: {type(data_obj)}")
    print(f"arr_obj type: {type(arr_obj)}")

    user = User(**data_obj)
    print(user)


class NameModel(BaseModel):
    name: str


class UserModel(BaseModel):
    id: int
    person: NameModel


def tool_pydantic():
    import json

    arr_str = (
        '[{"id": 1, "person": {"name": "er"}}, {"id": 2, "person": {"name": "san"}}]'
    )

    # 直接解析JSON
    users = [UserModel.model_validate(item) for item in json.loads(arr_str)]

    for u in users:
        print(f"ID: {u.id}, Name: {u.person.name}")


if __name__ == "__main__":
    tool_pydantic()
