def max_decker_cheeseburgers(A, B, C):
    n = A + B * 2
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        if i >= A:
            dp[i] = max(dp[i], dp[i - A] + 1)
        if i >= B * 2:
            dp[i] = max(dp[i], dp[i - B * 2] + 2)
    return dp[n]

def solve_problem():
    T = int(input())
    results = []
    for t in range(1, T + 1):
        A = int(input())
        B = int(input())
        C = int(input())
        result = max_decker_cheeseburgers(A, B, C)
        results.append(f'Case #{t}: {result}')
    for result in results:
        print(result)
    
if __name__ == "__main__":
    solve_problem()
