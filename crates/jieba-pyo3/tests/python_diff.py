#!/usr/bin/env python3
"""jieba_pyo3 与 Python jieba 的行为对齐 diff（P2 验收）。

硬断言：基础功能、自定义词典、add_word 生效。
软门槛：语料对齐率 >= 0.85（jieba-rs 与 CPython jieba 算法独立实现，
        极小比例分歧属预期，对齐率用于监控回归而非追求 100%）。
"""
import sys
import jieba
import jieba_pyo3

CORPUS = [
    "麦麦是最可爱的机器人，请多多关照",
    "我今天去了北京大学和清华大学",
    "自然语言处理是人工智能的重要方向",
    "他来到了网易杭研大厦",
    "李明硕士毕业于中国科学院计算所",
    "雷猴，麦麦你今天心情怎么样呀",
    "这个东西超级好用的说，安利给你",
    "我裂开了，这波操作太秀了",
    "上海市浦东新区张杨路500号",
    "2026年11月4日，天气晴朗，适合写代码",
    "性价比高，物流速度快，好评",
    "你不要过来啊！！！",
]

def main():
    total = agree = 0
    for text in CORPUS:
        expected = jieba.lcut(text)
        actual = jieba_pyo3.lcut(text)
        total += 1
        if expected == actual:
            agree += 1
        else:
            print(f"DIFF: {text}\n  jieba: {expected}\n  rust : {actual}")
    rate = agree / total
    print(f"\n对齐率: {agree}/{total} = {rate:.2%}")
    assert rate >= 0.85, f"对齐率 {rate:.2%} 低于门槛 85%"

    # 基础功能：模块级 + 实例级
    assert jieba_pyo3.lcut("麦麦是机器人") == jieba_pyo3.Jieba().lcut("麦麦是机器人")
    assert jieba_pyo3.cut("测试文本") == jieba_pyo3.lcut("测试文本")
    assert jieba_pyo3.lcut("麦麦", cut_all=True)

    # 自定义词典 + add_word
    j = jieba_pyo3.Jieba()
    with open("/tmp/user_dict_test.txt", "w") as f:
        f.write("# 测试词典\n麦麦机器人 1000\n")
    j.load_dict("/tmp/user_dict_test.txt")
    j.add_word("绞杀者模式", 100)
    segs = j.lcut("麦麦机器人启动了绞杀者模式")
    assert "麦麦机器人" in segs, segs
    assert "绞杀者模式" in segs, segs

    print("ALL ASSERTIONS PASSED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
