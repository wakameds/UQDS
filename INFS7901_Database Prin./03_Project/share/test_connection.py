import pymysql

def conn():
    mydb=pymysql.Connect('localhost','rischan','1234','rischandb',autocommit=True)
    return mydb.cursor()

def db_exe(query,c):
    try:
        if c.connection:
            print("connection exists")
            c.execute(query)
            return c.fetchall()
        else:
            print("trying to reconnect")
            c=conn()
    except Exception as e:
        return str(e)



dbc=conn()
data = db_exe("select * from phone_book",dbc)
data2 = db_exe("select * from phone_book where id=2",dbc)
data2 = db_exe("insert..",dbc)

print(data)