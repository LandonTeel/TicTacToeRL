import torch  

torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("Hey")

if __name__ == "__main__":
    main()
