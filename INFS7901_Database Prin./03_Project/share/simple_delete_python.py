import pymysql

con = pymysql.connect('localhost','rischan','1234','rischandb',autocommit=True)

with con: 

    cur = con.cursor()
    cur.execute("DELETE FROM phone_book WHERE id=1")

    rows = cur.fetchall()

    print(rows)