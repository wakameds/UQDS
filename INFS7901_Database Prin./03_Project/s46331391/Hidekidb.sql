-- phpMyAdmin SQL Dump
-- version 5.0.2
-- https://www.phpmyadmin.net/
--
-- Host: localhost
-- Generation Time: Jun 13, 2020 at 03:18 AM
-- Server version: 10.4.11-MariaDB
-- PHP Version: 7.4.4

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `Hidekidb`
--

-- --------------------------------------------------------

--
-- Table structure for table `DEPARTMENT`
--

CREATE TABLE `DEPARTMENT` (
  `dname` varchar(40) NOT NULL,
  `dlocation` varchar(40) NOT NULL,
  `dbuilding` varchar(40) NOT NULL,
  `dfloor` varchar(40) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `DEPARTMENT`
--

INSERT INTO `DEPARTMENT` (`dname`, `dlocation`, `dbuilding`, `dfloor`) VALUES
('Account', 'Kanagawa Office', 'HD1', '3F'),
('Product Development', 'Chiba Laboratory', 'DL2', '1F');

-- --------------------------------------------------------

--
-- Table structure for table `EMPLOYEE`
--

CREATE TABLE `EMPLOYEE` (
  `eid` int(11) NOT NULL,
  `ename` varchar(40) NOT NULL,
  `mail` varchar(40) NOT NULL,
  `dname` varchar(40) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `EMPLOYEE`
--

INSERT INTO `EMPLOYEE` (`eid`, `ename`, `mail`, `dname`) VALUES
(300003, 'Eri Waku', '12347@consumer.jp.com', 'Product Development'),
(300005, 'Koji Tate', '12349@consumer.jp.com', 'Factory'),
(300006, 'Maiko Wamura', '12310@consumer.jp.com', 'Product Development'),
(300007, 'Maiko Yuki', '12311@consumer.jp.com', 'Product Development'),
(300008, 'Masaru Shinoda', '12312@consumer.jp.com', 'Product Development'),
(300009, 'Mikie Nihei', '12313@consumer.jp.com', 'Product Development'),
(300010, 'Nana Toyonaga', '12314@consumer.jp.com', 'Product Development'),
(300011, 'NAri Wanda', '12315@consumer.jp.com', 'Factory'),
(300012, 'Noriyuki Shimada', '12361@consumer.jp.com', 'Factory'),
(300013, 'Shyuko Niho', '12317@consumer.jp.com', 'Factory'),
(300014, 'Take Hisamoto', '12318@consumer.jp.com', 'Factory'),
(300015, 'Tatsuo Matsuhahsi', '12319@consumer.jp.com', 'Factory'),
(300016, 'Taturo Shiono', '12320@consumer.jp.com', 'Human Resorce'),
(300017, 'Tomohiro Tamatsukiri', '12321@consumer.jp.com', 'Account'),
(300018, 'Toru Tsugane', '12322@consumer.jp.com', 'Account'),
(300019, 'Toshimune Michishige', '12323@consumer.jp.com', 'Marketing'),
(300020, 'Yuya Hoside', '12324@consumer.jp.com', 'Human Resorce'),
(326301, 'Hideki Wakayama', 'wakahide@kome.jp', 'Research & Development'),
(333333, 'aoki', 'aksjd', 'Salesman');

-- --------------------------------------------------------

--
-- Table structure for table `EMPLOYEE-PHONE`
--

CREATE TABLE `EMPLOYEE-PHONE` (
  `eid` int(11) NOT NULL,
  `phone` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `EMPLOYEE-PHONE`
--

INSERT INTO `EMPLOYEE-PHONE` (`eid`, `phone`) VALUES
(300003, 902284113),
(300005, 902284115),
(300006, 902284116),
(300007, 902284132),
(300008, 902284118),
(300009, 902284119),
(300010, 902284133),
(300011, 902284121),
(300012, 902284122),
(300013, 902284134),
(300014, 902284124),
(300015, 902284125),
(300016, 902284126),
(300017, 902284127),
(300018, 902284128),
(300019, 902284129),
(300020, 902284130);

-- --------------------------------------------------------

--
-- Table structure for table `EMPLOYEE_ADDRESS`
--

CREATE TABLE `EMPLOYEE_ADDRESS` (
  `eid` int(11) NOT NULL,
  `postcode` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `EMPLOYEE_ADDRESS`
--

INSERT INTO `EMPLOYEE_ADDRESS` (`eid`, `postcode`) VALUES
(300015, 1500021),
(300014, 1500022),
(300019, 1500033),
(300007, 1506004),
(300008, 1506005),
(300009, 1506006),
(300010, 1506007),
(300011, 1506008),
(300012, 1506009),
(300003, 1506036),
(300005, 1506038),
(300006, 1506039),
(300013, 1506090),
(300016, 1650022),
(300017, 1650023),
(300018, 1680064),
(300020, 2891701);

-- --------------------------------------------------------

--
-- Table structure for table `EMPLOYEE_GRADE`
--

CREATE TABLE `EMPLOYEE_GRADE` (
  `eid` int(11) NOT NULL,
  `grade` int(11) NOT NULL,
  `year` year(4) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `EMPLOYEE_GRADE`
--

INSERT INTO `EMPLOYEE_GRADE` (`eid`, `grade`, `year`) VALUES
(300003, 1, 2018),
(300005, 5, 2017),
(300006, 3, 2016),
(300007, 3, 2016),
(300008, 3, 2016),
(300009, 4, 2018),
(300010, 5, 2018),
(300011, 5, 2018),
(300012, 6, 2018),
(300013, 6, 2019),
(300014, 4, 2019),
(300015, 2, 2019),
(300016, 1, 2019),
(300017, 1, 2019),
(300018, 1, 2019),
(300019, 1, 2019),
(300020, 1, 2019);

-- --------------------------------------------------------

--
-- Table structure for table `PRODUCT`
--

CREATE TABLE `PRODUCT` (
  `pid` varchar(40) NOT NULL,
  `pname` varchar(40) NOT NULL,
  `category` varchar(40) NOT NULL,
  `price$` float NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `PRODUCT`
--

INSERT INTO `PRODUCT` (`pid`, `pname`, `category`, `price$`) VALUES
('DEO2001-01', 'Axe', 'deodrant', 10.4),
('DEO2001-02', 'Deo for men', 'deodrant', 6.6),
('DEO2001-03', 'Hygine', 'deodrant', 3),
('DEO2001-04', 'Neo', 'deodrant', 18),
('LD-2010-01', 'Hypper', 'landry', 10),
('LD-2010-04', 'Hypper2', 'landry', 450),
('LD-2010-05', 'Hypper3', 'landry', 500),
('LD1001-01', 'Atttack', 'landry', 18.6),
('LD1001-02', 'Ultra', 'landry', 11),
('LD1001-03', 'Hygine', 'landry', 5),
('LD1001-04', 'fafa', 'landry', 3.6),
('LD1001-05', 'Miracle', 'landry', 11),
('SK1001-01', 'beauty cosme', 'skincare', 50.1),
('SK1001-02', 'silk', 'skincare', 20.5),
('SK1001-03', '7in1', 'skincare', 100.1),
('SP3001-01', 'Harbarom', 'shampoo', 22.3),
('SP3001-02', 'Nutumage', 'shampoo', 18.8),
('SP3001-03', 'Ninag', 'shampoo', 19),
('SP3001-04', 'Opo', 'shampoo', 22),
('SP3001-05', 'MAKE', 'shampoo', 10.5);

-- --------------------------------------------------------

--
-- Table structure for table `PRODUCT_SALE`
--

CREATE TABLE `PRODUCT_SALE` (
  `pid` varchar(40) NOT NULL,
  `year` year(4) NOT NULL,
  `amount$k` float NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `PRODUCT_SALE`
--

INSERT INTO `PRODUCT_SALE` (`pid`, `year`, `amount$k`) VALUES
('LD1001-01', 2006, 200),
('LD1001-01', 2007, 600),
('LD1001-01', 2008, 1000),
('LD1001-01', 2009, 2000),
('LD1001-01', 2010, 3500),
('LD1001-01', 2011, 2000),
('LD1001-01', 2012, 1600),
('LD1001-01', 2013, 6000),
('LD1001-01', 2014, 5000),
('LD1001-01', 2015, 7000),
('LD1001-01', 2016, 8000),
('LD1001-01', 2017, 10000),
('LD1001-01', 2018, 7500),
('SK1001-01', 2010, 1000),
('SK1001-01', 2011, 1300),
('SK1001-01', 2012, 1400),
('SK1001-01', 2013, 1600),
('SK1001-01', 2014, 2500),
('SK1001-01', 2015, 4000),
('SK1001-01', 2016, 8000);

-- --------------------------------------------------------

--
-- Table structure for table `PROJECT`
--

CREATE TABLE `PROJECT` (
  `proid` varchar(40) NOT NULL,
  `proname` varchar(255) NOT NULL,
  `profield` varchar(40) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `PROJECT`
--

INSERT INTO `PROJECT` (`proid`, `proname`, `profield`) VALUES
('DVSKP001', 'developing technology for skincare', 'skincare'),
('FSKP001', 'expanding production of skincare products', 'skincare'),
('HRP001', 'enhance human resource project', 'human resource'),
('MKLDP001', 'marketing of landry project', 'landry'),
('MKSKP001', 'marketing of skincare project', 'skincare'),
('SCMP001', 'effective supply chain', 'supply chain');

-- --------------------------------------------------------

--
-- Table structure for table `PROJECT_SCHEDULE`
--

CREATE TABLE `PROJECT_SCHEDULE` (
  `proid` varchar(40) NOT NULL,
  `situation` varchar(40) NOT NULL,
  `prostart` date NOT NULL,
  `proend` date NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `PROJECT_SCHEDULE`
--

INSERT INTO `PROJECT_SCHEDULE` (`proid`, `situation`, `prostart`, `proend`) VALUES
('DVSKP001', 'proceed', '2017-10-17', '0000-00-00'),
('FSKP001', 'proceed', '2018-11-18', '0000-00-00'),
('HRP001', 'proceed', '2019-01-19', '0000-00-00'),
('MKLDP001', 'finish', '2019-04-16', '2020-04-17'),
('MKSKP001', 'proceed', '2019-01-15', '0000-00-00'),
('SCMP001', 'proceed', '2019-09-29', '0000-00-00');

-- --------------------------------------------------------

--
-- Table structure for table `WAREHOUSE`
--

CREATE TABLE `WAREHOUSE` (
  `wid` varchar(40) NOT NULL,
  `wname` varchar(40) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `WAREHOUSE`
--

INSERT INTO `WAREHOUSE` (`wid`, `wname`) VALUES
('K001', 'Kagawa warehouse1'),
('K002', 'Kagawa warehouse2'),
('O001', 'Osaka warehouse1'),
('O002', 'Osaka warehouse2'),
('T001', 'Tokyo warehouse1'),
('T002', 'Tokyo warehouse2');

-- --------------------------------------------------------

--
-- Table structure for table `WAREHOUSE_PRODUCT`
--

CREATE TABLE `WAREHOUSE_PRODUCT` (
  `pid` varchar(40) NOT NULL,
  `wid` varchar(40) NOT NULL,
  `stdate` date NOT NULL,
  `stamountcount` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `WAREHOUSE_PRODUCT`
--

INSERT INTO `WAREHOUSE_PRODUCT` (`pid`, `wid`, `stdate`, `stamountcount`) VALUES
('DEO2001-01', 'T001', '2020-02-02', 2000),
('DEO2001-02', 'T001', '2020-02-03', 20000),
('SK1001-01', 'T001', '2020-01-30', 2),
('SK1001-02', 'O001', '2020-01-31', 3),
('SK1001-02', 'T001', '2020-02-04', 300),
('SK1001-03', 'O002', '2020-02-01', 1000);

-- --------------------------------------------------------

--
-- Table structure for table `WORK_PRODUCT`
--

CREATE TABLE `WORK_PRODUCT` (
  `eid` int(11) NOT NULL,
  `pid` varchar(40) NOT NULL,
  `hour` float NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `WORK_PRODUCT`
--

INSERT INTO `WORK_PRODUCT` (`eid`, `pid`, `hour`) VALUES
(300003, 'SK1001-02', 340),
(300003, 'SP3001-02', 400);

-- --------------------------------------------------------

--
-- Table structure for table `WORK_PROJECT`
--

CREATE TABLE `WORK_PROJECT` (
  `eid` int(11) NOT NULL,
  `proid` varchar(40) NOT NULL,
  `hour` float NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `WORK_PROJECT`
--

INSERT INTO `WORK_PROJECT` (`eid`, `proid`, `hour`) VALUES
(300003, 'DVSKP001', 3949),
(300005, 'DVSKP001', 34),
(300005, 'MKLDP001', 23),
(300006, 'DVSKP001', 23),
(300007, 'MKSKP001', 445),
(300008, 'MKLDP001', 431);

--
-- Indexes for dumped tables
--

--
-- Indexes for table `DEPARTMENT`
--
ALTER TABLE `DEPARTMENT`
  ADD PRIMARY KEY (`dname`),
  ADD KEY `d.name` (`dname`);

--
-- Indexes for table `EMPLOYEE`
--
ALTER TABLE `EMPLOYEE`
  ADD PRIMARY KEY (`eid`),
  ADD KEY `d.name` (`dname`);

--
-- Indexes for table `EMPLOYEE-PHONE`
--
ALTER TABLE `EMPLOYEE-PHONE`
  ADD PRIMARY KEY (`eid`),
  ADD KEY `e.id` (`eid`);

--
-- Indexes for table `EMPLOYEE_ADDRESS`
--
ALTER TABLE `EMPLOYEE_ADDRESS`
  ADD PRIMARY KEY (`eid`),
  ADD KEY `e.id` (`eid`),
  ADD KEY `postcode` (`postcode`);

--
-- Indexes for table `EMPLOYEE_GRADE`
--
ALTER TABLE `EMPLOYEE_GRADE`
  ADD PRIMARY KEY (`eid`,`year`),
  ADD KEY `e.id` (`eid`);

--
-- Indexes for table `PRODUCT`
--
ALTER TABLE `PRODUCT`
  ADD PRIMARY KEY (`pid`);

--
-- Indexes for table `PRODUCT_SALE`
--
ALTER TABLE `PRODUCT_SALE`
  ADD PRIMARY KEY (`pid`,`year`),
  ADD KEY `p.id` (`pid`);

--
-- Indexes for table `PROJECT`
--
ALTER TABLE `PROJECT`
  ADD PRIMARY KEY (`proid`);

--
-- Indexes for table `PROJECT_SCHEDULE`
--
ALTER TABLE `PROJECT_SCHEDULE`
  ADD PRIMARY KEY (`proid`);

--
-- Indexes for table `WAREHOUSE`
--
ALTER TABLE `WAREHOUSE`
  ADD PRIMARY KEY (`wid`);

--
-- Indexes for table `WAREHOUSE_PRODUCT`
--
ALTER TABLE `WAREHOUSE_PRODUCT`
  ADD PRIMARY KEY (`pid`,`wid`),
  ADD KEY `p.id` (`pid`,`wid`),
  ADD KEY `from warehouse to warehouse` (`wid`);

--
-- Indexes for table `WORK_PRODUCT`
--
ALTER TABLE `WORK_PRODUCT`
  ADD PRIMARY KEY (`eid`,`pid`),
  ADD KEY `e.id` (`eid`),
  ADD KEY `pro.id` (`pid`);

--
-- Indexes for table `WORK_PROJECT`
--
ALTER TABLE `WORK_PROJECT`
  ADD PRIMARY KEY (`eid`,`proid`),
  ADD KEY `e.id` (`eid`),
  ADD KEY `pro.id` (`proid`);

--
-- Constraints for dumped tables
--

--
-- Constraints for table `DEPARTMENT`
--
ALTER TABLE `DEPARTMENT`
  ADD CONSTRAINT `d.name from employee` FOREIGN KEY (`dname`) REFERENCES `EMPLOYEE` (`dname`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `EMPLOYEE-PHONE`
--
ALTER TABLE `EMPLOYEE-PHONE`
  ADD CONSTRAINT `from employee` FOREIGN KEY (`eid`) REFERENCES `EMPLOYEE` (`eid`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `EMPLOYEE_ADDRESS`
--
ALTER TABLE `EMPLOYEE_ADDRESS`
  ADD CONSTRAINT `from employee_adress` FOREIGN KEY (`eid`) REFERENCES `EMPLOYEE` (`eid`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `EMPLOYEE_GRADE`
--
ALTER TABLE `EMPLOYEE_GRADE`
  ADD CONSTRAINT `from employee_grade` FOREIGN KEY (`eid`) REFERENCES `EMPLOYEE` (`eid`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `PRODUCT_SALE`
--
ALTER TABLE `PRODUCT_SALE`
  ADD CONSTRAINT `from product` FOREIGN KEY (`pid`) REFERENCES `PRODUCT` (`pid`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `PROJECT_SCHEDULE`
--
ALTER TABLE `PROJECT_SCHEDULE`
  ADD CONSTRAINT `situation_pro.id` FOREIGN KEY (`proid`) REFERENCES `PROJECT` (`proid`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `WAREHOUSE_PRODUCT`
--
ALTER TABLE `WAREHOUSE_PRODUCT`
  ADD CONSTRAINT `from prduct to warehouse` FOREIGN KEY (`pid`) REFERENCES `PRODUCT` (`pid`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `from warehouse to warehouse` FOREIGN KEY (`wid`) REFERENCES `WAREHOUSE` (`wid`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `WORK_PRODUCT`
--
ALTER TABLE `WORK_PRODUCT`
  ADD CONSTRAINT `from employee to work_pro` FOREIGN KEY (`eid`) REFERENCES `EMPLOYEE` (`eid`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `from product to work_pro` FOREIGN KEY (`pid`) REFERENCES `PRODUCT` (`pid`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `WORK_PROJECT`
--
ALTER TABLE `WORK_PROJECT`
  ADD CONSTRAINT `from project to workpro` FOREIGN KEY (`proid`) REFERENCES `PROJECT` (`proid`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `from wmplo to workpro` FOREIGN KEY (`eid`) REFERENCES `EMPLOYEE` (`eid`) ON DELETE CASCADE ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
