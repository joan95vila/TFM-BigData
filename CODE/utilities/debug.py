def information_block(title, *args, version=1):
    if version == 1:
        print(f"\n\n{title}\n{'=' * 150}")
        for arg in args:
            print(f"{arg}")
        print(f"{'=' * 150}")
    elif version == 2:
        print(f"\n{'=' * 150}\n|{' ' * int((150 - len(title)) / 2)}{title}\n|{'-' * 149}")
        for arg in args:
            print(f"|\t>> {arg}")
        print(f"{'=' * 150}")