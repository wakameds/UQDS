import pymysql

con = pymysql.connect('localhost','rischan','1234','rischandb',autocommit=True)

with con: 

    cur = con.cursor()
    cur.execute("SELECT * FROM phone_book")

    rows = cur.fetchall()

    print(rows)