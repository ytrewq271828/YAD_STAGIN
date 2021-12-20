def trace(func):
    def wrapper():
        print(func.__name__, "begins")
        func()
        print(func.__name__, "ends")
        return
    return wrapper