class Foo:

    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.n == 19:
            raise StopIteration
        self.n += 1
        return self.n


f1 = Foo(10)
# print(f1.__next__())
# print(f1.__next__())
# print(f1.__next__())
# print(f1.__next__())

# 迭代器协议之一：必须要有__iter__方法，之二：必须要有__next__方法
# 当遇到抛出StopIteration异常时，结束循环
for i in f1:  # f1.__iter__()  == iter(f1)
    print(i)