import pandas


def removeTrueNegative(positive_train_file, true_negative_train_file):



if __name__ == '__main__':
    parser = OptionParser()
    # Read the options
    parser.add_option("", "--positive-train-file", dest="positive_train_file", default="", help="")
    parser.add_option("", "--true-negative-file", dest="true_negative_train_file", default="",help="")

    (options, args) = parser.parse_args()

    positive_train_file = options.positive_train_file
    true_negative_train_file = options.true_negative_train_file

