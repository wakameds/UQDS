import pymysql

class Database:
    def connect(self):
        return pymysql.connect(host='localhost',
                             user='Hideki',
                             password='0287Waka',
                             db='Hidekidb')


    def read(self, id):
        con = Database.connect(self)
        cursor = con.cursor()
        try:
            if id == None:
                cursor.execute("SELECT * FROM EMPLOYEE order by ename asc")
            else:
                cursor.execute("SELECT * FROM EMPLOYEE where eid = %s order by ename asc", (id,))
            return cursor.fetchall()

        except:
            return ()
        finally:
            con.close()

####EMPLOYEE####
################
    def read_employee(self, id):
        con = Database.connect(self)
        cursor = con.cursor()
        try:
            if id == None:
                cursor.execute("SELECT * FROM EMPLOYEE order by ename asc")
            else:
                cursor.execute("SELECT * FROM EMPLOYEE where eid = %s order by ename asc", (id,))
            return cursor.fetchall()
        except:
            return ()
        finally:
            con.close()


    def employee_grade(self, id):
        con = Database.connect(self)
        cursor = con.cursor()
        try:
            if id == None:
                cursor.execute("SELECT A.eid, B.ename, A.grade, A.year FROM EMPLOYEE_GRADE A, EMPLOYEE B WHERE A.eid = B.eid ORDER BY A.year ASC")
            else:
                cursor.execute("SELECT A.eid, B.ename, A.grade, A.year FROM EMPLOYEE_GRADE A, EMPLOYEE B WHERE A.eid = B.eid AND A.eid = %s ORDER BY A.year ASC", (id,))
            return cursor.fetchall()
        except:
            return ()
        finally:
            con.close()

    def insert(self,data):
        con = Database.connect(self)
        cursor = con.cursor()
        try:
            cursor.execute("INSERT INTO EMPLOYEE(eid,ename,mail,dname) VALUES(%s,%s,%s,%s)",(data['eid'],data['ename'],data['mail'],data['dname'],))
            con.commit()
            return True
        except:
            con.rollback()
            return False
        finally:
            con.close()

    def employee_update(self, id, data):
        con = Database.connect(self)
        cursor = con.cursor()
        try:
            cursor.execute("UPDATE EMPLOYEE set eid = %s,  grade = %s where eid = %s", (data['eid'],data['ename'],data['mail'],data['dname'],id,))
            con.commit()
            return True
        except:
            con.rollback()
            return False
        finally:
            con.close()

    def deleteemployee(self, id):
        con = Database.connect(self)
        cursor = con.cursor()
        try:
            cursor.execute("DELETE FROM EMPLOYEE where eid = %s", (id,))
            con.commit()
            return True
        except:
            con.rollback()
            return False
        finally:
            con.close()

    def update(self, id, data):
        con = Database.connect(self)
        cursor = con.cursor()
        try:
            cursor.execute("UPDATE EMPLOYEE set eid = %s, ename = %s, mail = %s, dname = %s where eid = %s", (data['eid'],data['ename'],data['mail'],data['dname'],id,))
            con.commit()
            return True
        except:
            con.rollback()
            return False
        finally:
            con.close()

    def delete(self, id):
        con = Database.connect(self)
        cursor = con.cursor()
        try:
            cursor.execute("DELETE FROM EMPLOYEE where eid = %s", (id,))
            con.commit()
            return True
        except:
            con.rollback()
            return False
        finally:
            con.close()


####PRODUCT####
################
    def read_ploduct(self, id):
        con = Database.connect(self)
        cursor = con.cursor()
        try:
            if id == None:
                cursor.execute("SELECT * FROM PRODUCT order by pname asc")
            else:
                cursor.execute("SELECT * FROM PRODUCT where pid = %s order by pname asc", (id,))
            return cursor.fetchall()
        except:
            return ()
        finally:
            con.close()




####PROJECT####
################
    def read_project(self, id):
        con = Database.connect(self)
        cursor = con.cursor()
        try:
            if id == None:
                cursor.execute("SELECT P.proid, P.proname, P.profield, P2.prostart, P2.proend, P2.situation FROM PROJECT P, PROJECT_SCHEDULE P2 WHERE P.proid = P2.proid ORDER BY P2.prostart DESC")
            else:
                cursor.execute("SELECT P.proid, P.proname, P.profield, P2.prostart, P2.proend, P2.situation FROM PROJECT P, PROJECT_SCHEDULE P2 WHERE P.proid = P2.proid AND P.proid = %s ORDER BY P2.prostart DESC", (id,))
            return cursor.fetchall()
        except:
            return ()
        finally:
            con.close()


    def project_teams(self, id):
        con = Database.connect(self)
        cursor = con.cursor()
        try:
            if id == None:
                cursor.execute("SELECT P.proid, P.proname, count(WP.eid), sum(WP.hour) FROM PROJECT P INNER JOIN WORK_PROJECT WP ON P.proid = WP.proid GROUP BY P.proid")
            else:
                cursor.execute("SELECT P.proid, P.proname, count(WP.eid), sum(WP.hour) FROM PROJECT P INNER JOIN WORK_PROJECT WP ON P.proid = WP.proid AND P.proid = %s GROUP BY P.proid")
            return cursor.fetchall()
        except:
            return ()
        finally:
            con.close()

