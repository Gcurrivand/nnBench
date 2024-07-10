from image_processing import binarize_image, binarize_folder

def main():
    print("Which function do you want to run?")
    print("1. Binarize folder(Dataset/Numbers, Dataset/NumbersFC,250)")

    while True:
        try:
            choice = int(input("Enter the number of the function (1-3): "))
            if choice == 1:
                binarize_folder("Dataset/Numbers", "Dataset/NumbersFC",250)
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    main()
