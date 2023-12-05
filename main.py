import libraries.create as create

try:
    if __name__ == "__main__":
        print("welcome to the drone object detector!")
        selection = input(
            "\nwhat would you like to do?\n1. create a new model\n2. train an existing model\n3. test an existing model\n4. exit\n"
        )

        if selection == "1":
            selection = input(
                "\nwould you like to set the model's parameters manually? (y/n)\n"
            )

            if selection == "y":
                epochs = int(input("\nhow many epochs would you like to train for?\n"))
                if epochs <= 0:
                    raise ValueError

                if epochs > 500:
                    selection = input(
                        "WARNING: training for more than 500 epochs may cause overfitting.\nwould you like to continue? (y/n)\nchoice: "
                    )
                    if selection == "n":
                        exit()

                batch_size = int(input("what batch size would you like to use?\n"))
                if batch_size < 1:
                    raise ValueError

                learning_rate = float(
                    input("what learning rate would you like to use? (0 - 1)\n")
                )
                if learning_rate < 0 or learning_rate > 1:
                    raise ValueError

                optimizer = input(
                    "what optimizer would you like to use? (SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto)\n"
                )
                choices = [
                    "SGD",
                    "Adam",
                    "Adamax",
                    "AdamW",
                    "NAdam",
                    "RAdam",
                    "RMSProp",
                    "auto",
                ]
                if optimizer not in choices:
                    raise ValueError

                device = input(
                    "what device would you like to use? (0 = gpu, cpu = cpu, mps = apple silicon gpu)\n"
                )
                print(device)
                if device != "0" and device != "cpu" and device != "mps":
                    raise ValueError

                create.main(epochs, batch_size, learning_rate, optimizer, device)

            elif selection == "n":
                create.main()

            else:
                raise ValueError

        elif selection == "2":
            print("this feature is not yet implemented. please try again later.")
            exit()

        elif selection == "3":
            print("this feature is not yet implemented. please try again later.")
            exit()

        elif selection == "4":
            print("exiting...")
            exit()
            
except Exception as e:
    print(f"an error occurred: {e}")
    exit()