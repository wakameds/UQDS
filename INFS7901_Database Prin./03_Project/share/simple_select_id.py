import pymysql

con = pymysql.connect('localhost','rischan','1234','rischandb',autocommit=True)

id_var = None

with con: 

    cur = con.cursor()

    if id_var == None:
    	cur.execute("SELECT * FROM phone_book")
    else:
    	cur.execute("SELECT name, phone FROM phone_book WHERE id = %s ", id_var)

    rows = cur.fetchall()

    print(rows)