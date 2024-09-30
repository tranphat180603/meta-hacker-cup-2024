import math
def calculate_p_increase(N, P):
    P_decimal = P / 100
    P_prime = 100 * math.pow(P_decimal, (N-1)/N)
    return P_prime - P

def main():
    T = int(input())
    results = []
    for i in range(1, T+1):
        N, P = map(int, input().split())
        result = calculate_p_increase(N, P)
        results.append(f"Case #{i}: {result}")
    for result in results:
        print(result)

if __name__ == "__main__":
    main()