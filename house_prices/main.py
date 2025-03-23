# main.py


import utils_io

path_train_data = "data_set/train.csv"
path_test_data = "data_set/test.csv"


def main():
    print("Hello World")
    train_data = utils_io.load_csv_data(path_train_data)
    test_data = utils_io.load_csv_data(path_test_data)

    # Print the loaded data
    print(train_data.head)
    print(train_data.shape)


if __name__ == "__main__":
    main()
