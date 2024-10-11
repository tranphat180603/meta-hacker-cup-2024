def solve_bridge_crossing():
    import sys

    # Read all input lines until EOF
    tokens = []
    try:
        while True:
            line = input()
            if line.strip() == '':
                continue  # Skip empty lines
            tokens += line.strip().split()
    except EOFError:
        pass  # No more input

    ptr = 0  # Pointer to traverse the tokens

    # Read the number of test cases
    if ptr >= len(tokens):
        sys.exit("No input provided.")
    T = int(tokens[ptr])
    ptr += 1

    for test_case in range(1, T + 1):
        # Read N and K
        if ptr + 1 >= len(tokens):
            print(f"Case #{test_case}: NO")
            ptr += 2  # Attempt to skip to next test case
            continue

        N = int(tokens[ptr])
        K = int(tokens[ptr + 1])
        ptr += 2

        # Read S_i
        if ptr + N > len(tokens):
            print(f"Case #{test_case}: NO")
            ptr += N  # Attempt to skip to next test case
            continue

        S = list(map(int, tokens[ptr:ptr + N]))
        ptr += N

        # Sort the crossing times in ascending order
        S.sort()

        total_time = 0  # Initialize total crossing time
        n = N  # Number of travelers remaining to cross

        while n > 0:
            if n == 1:
                # Only one traveler left; they cross alone
                total_time += S[0]
                n -= 1
            elif n == 2:
                # Two travelers; both cross together
                total_time += S[0]
                n -= 2
            else:
                # More than two travelers
                # Step 1: Fastest takes the slowest across
                total_time += S[0]
                # Step 2: Fastest returns to ferry the next traveler
                total_time += S[0]
                n -= 1  # One traveler has crossed

        # Determine if all travelers can cross within K seconds
        if total_time <= K:
            result = "YES"
        else:
            result = "NO"

        # Output the result for the current test case
        print(f"Case #{test_case}: {result}")

# Execute the function when the script is run
if __name__ == "__main__":
    solve_bridge_crossing()
