import argparse
import os

def main():
    file_list = os.listdir("/export/b15/haoranxu/clce/fasttext/")
    records = set()
    num = 0

    f_w = open("/export/b15/haoranxu/clce/fasttext/vocab.txt", "w", encoding="utf-8")
    f_w.writelines(["@@UNKNOWN@@", "\n"])
    for file in file_list:
        with open("/export/b15/haoranxu/clce/fasttext/" + file, encoding="utf-8") as f:
            line = f.readline()
            while (line):
                line = line.split()
                if len(line) > 10 and line[0] not in records:
                    word = line[0]
                    records.add(word)
                    f_w.writelines([word,"\n"])
                    num += 1
                    if num % 10000 ==0 :
                        print(num)
                line = f.readline()

    f_w.close()

if __name__ == '__main__':
    main()