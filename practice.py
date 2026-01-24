def print_mario(height):
    for row in range(1, height + 1):
        print(("#" * row),("#" * row)) 


def main():
    while True:
        try:
            height = int(input("Height: "))
            if height > 0:
                break
        except ValueError:
            pass

    print_mario(height)

if __name__ == "__main__":
    main()