def plus(n1, n2): return (n1 + n2)

def minus(n1, n2): return (n1 - n2)

def multiply (n1, n2): return (n1 * n2)

def divide (n1, n2): return (n1 / n2)
    
def calculator():
    """로봇 SW 개발 입문용 계산기 메인 함수"""
    while True:
        try:
            op = input("choose one +-*/q: ")

            if op == 'q':
                break

            if op not in ['+', '-', '*', '/']:
                continue
            
            n1 = float(input("1st number: "))
            n2 = float(input("2nd number: "))
            
            if op == '+':
                print(f"{n1} + {n2} = {plus(n1, n2)}")
            elif op == '-':
                print(f"{n1} - {n2} = {minus(n1, n2)}")
            elif op == '*':
                print(f"{n1} * {n2} = {multiply(n1, n2)}")
            elif op == '/':
                print(f"{n1} / {n2} = {divide(n1, n2)}")
        except ValueError as e:
            print(e)
        except ZeroDivisionError as e:
            print(e)
        except Exception as e:
            print(e)           

# 외부에서 import 했을때 현재 파일이 메인파일이 아닌경우는 이 파일을 실행하지 않는다
if __name__ == "__main__":
    calculator()
