import matplotlib.pyplot as plt
from scipy import stats

# RUIDO EN LOS SENSORES
# # Agente 2
# fitness0 = [0.32, 0.459, 0.37, 0.33, 0.409, 0.274, 0.4667, 0.3879, 0.4511, 0.40, 0.458, 0.4926, 0.488, 0.293, 0.327, 0.448, 0.302, 0.3968, 0.383, 0.386, 0.36, 0.32, 0.38, 0.374, 0.371, 0.3418, 0.358, 0.348, 0.5055298, 0.35238]
# fitness02 = [0.32, 0.325, 0.342, 0.37, 0.4018, 0.283, 0.34438, 0.4644, 0.291, 0.377, 0.3129, 0.3828, 0.405, 0.418, 0.288, 0.417, 0.359, 0.266, 0.326, 0.318, 0.5463, 0.379046, 0.373, 0.313, 0.41048, 0.358, 0.521035, 0.339, 0.3055, 0.3214]
# fitness034 = [0.377, 0.44837, 0.3603, 0.2666, 0.362, 0.30, 0.277, 0.429348, 0.4659, 0.365, 0.31258, 0.326, 0.426, 0.37946, 0.338, 0.34799, 0.29072, 0.3658, 0.4572, 0.34, 0.42, 0.4528, 0.384, 0.379, 0.461, 0.270, 0.43837, 0.4588, 0.387, 0.3928]
# fitness044 = [0.3887, 0.325, 0.298456, 0.416, 0.400, 0.377, 0.2958, 0.3479, 0.43636, 0.3706, 0.4477379, 0.3479, 0.3007, 0.3854, 0.3223, 0.268186, 0.2703265, 0.39716, 0.3211, 0.41242, 0.339776, 0.4258, 0.439584, 0.39859, 0.3909, 0.3696, 0.3729369, 0.502196, 0.40348637, 0.31094]

# Agente 1
# fitness0 = [0.27, 0.2668, 0.176, 0.2356, 0.2732, 0.20, 0.16444, 0.2329, 0.284348, 0.2582887, 0.2310, 0.18948, 0.284434, 0.2582, 0.28744028, 0.3146814, 0.24239, 0.3314446, 0.19412, 0.163741, 0.1621, 0.2598, 0.1889, 0.1705, 0.2367, 0.2505, 0.2038, 0.1910, 0.2352, 0.2715399]
# fitness02 = [0.27296, 0.2112, 0.290427, 0.28656715, 0.137484, 0.1928, 0.2685, 0.2661, 0.29848, 0.2657, 0.206896, 0.203095, 0.25979,0.19666,0.288097,0.18379,0.241524,0.270378,0.28686,0.3442162,0.31055,0.37802,0.234,0.187,0.3121,0.248,0.3435,0.2859,0.2668,0.2035]
# fitness034 = [0.2651,0.3238,0.2202,0.327,0.223359,0.1475,0.2037,0.30769,0.28416,0.2149,0.32607,0.155373,0.18919,0.18871,0.2860,0.25838,0.204449,0.3060,0.317518,0.21799,0.18646,0.26936,0.269255,0.26839355,0.2818511,0.2170,0.2068,0.268994,0.21150,0.2106069]
# fitness044 = [0.29089,0.217027,0.1907,0.216621,0.189417,0.286810,0.256111,0.2697,0.361954,0.25836,0.2011,0.1967,0.132618,0.258237,0.23557,0.17934,0.16687,0.255864,0.26969,0.234474,0.2187,0.31405,0.29384,0.24508,0.33243,0.212608,0.311887,0.19168,0.221069,0.25309]

# RUIDO EN LOS RITMOS DE PLASTICIDAD
# # Agente 1
# fitness0 = [0.27, 0.2668, 0.176, 0.2356, 0.2732, 0.20, 0.16444, 0.2329, 0.284348, 0.2582887, 0.2310, 0.18948, 0.284434, 0.2582, 0.28744028, 0.3146814, 0.24239, 0.3314446, 0.19412, 0.163741, 0.1621, 0.2598, 0.1889, 0.1705, 0.2367, 0.2505, 0.2038, 0.1910, 0.2352, 0.2715399]
# fitness02 = [0.13, 0.19, 0.271, 0.169, 0.2753, 0.157015, 0.31034, 0.214506, 0.13678, 0.1616, 0.1598, 0.3063, 0.3019, 0.1109, 0.12808, 0.3068, 0.22629695, 0.18235, 0.1827, 0.1795, 0.2368, 0.2407, 0.238, 0.2071, 0.13498, 0.2395, 0.279, 0.30376, 0.2206, 0.219257]
# fitness034 = [0.12909803036420098,0.3033835550552459,0.19383175269558064,0.23939860563294374,0.23688232068265072,0.1858752833047749,0.22067344516642473,0.1682668315810135,0.12382328536400963,0.2306780963318457,0.23209439936683382,0.31675769287756006,0.16626496080833164,0.1693276242379106,0.26294012757737995,0.22492348507825616,0.20390328428280896,0.21323937442191396,0.25921261571409354,0.23827891182443223,0.20350777935078668,0.2292804036758912,0.20093812671980807,0.21682467434764588,0.2811562103489246,0.19786240356255957,0.28568682486547,0.3190673769890572,0.26473250285384586,0.272141080678051]
# fitness044 = [0.30149228552370944,0.1622215561625739,0.23427136131370663,0.33045049275602345,0.2233240246429865,0.1831672907972861,0.1997796800903175,0.2830814719871413,0.2616573000480283,0.18373691844196785,0.244948088740103,0.22662971795059625,0.2715009298956863,0.19957352406603418,0.31669654670828923,0.1819538921257141,0.26306220415862125,0.3068971676766033,0.3261143303171479,0.16898539222254005,0.23051585097852947,0.24559953700129508,0.31836302481265405,0.36214940228581144,0.24047731800563796,0.23817100822456128,0.29299884437976664,0.24851088417909872,0.25207257090911134,0.1799630624665923]

# Agente 2
fitness0 = [0.32, 0.459, 0.37, 0.33, 0.409, 0.274, 0.4667, 0.3879, 0.4511, 0.40, 0.458, 0.4926, 0.488, 0.293, 0.327, 0.448, 0.302, 0.3968, 0.383, 0.386, 0.36, 0.32, 0.38, 0.374, 0.371, 0.3418, 0.358, 0.348, 0.5055298, 0.35238]
fitness02 = [0.26665322496007143,0.38585025375347387,0.26452059696233204,0.1531420819712173,0.3261361998411274,0.4603354833744706,0.3690869029402329,0.31658011520800533,0.3584458240579993,0.4264535991656399,0.35015502440641355,0.32130010527009667,0.1938377896222617,0.3113260427249281,0.40181277048225494,0.32238788287962,0.2621264617432751,0.3138120008344637,0.3232959529080607,0.3712144961248677,0.3517865674758656,0.3677511131733229,0.4200608829880138,0.1937467371706107,0.21752098138297253,0.441646335920297,0.28622010779342916,0.22886271064981123,0.1644581193059148,0.4721375397064416]
fitness034 = [0.39424602745182746,0.3325341205690578,0.1816356097042361,0.3413746909164033,0.3367207094978798,0.38587948036654984,0.05136473862634361,0.358154246256004,0.24903212324147667,0.35729649277895076,0.4296128180026407,0.3825032682454985,0.3750278440152808,0.42633806991946144,0.4093697638113425,0.13054137443310831,0.4469776524241395,0.2289348489477107,0.32213913575990066,0.3300020202615595,0.3953899044259133,0.37864179679223897,0.3501039256180682,0.42595526635574316,0.4957360608345649,0.3442028174228953,0.06278128538974038,0.42753305730464986,0.3228265296319049,0.4034039568291313]
fitness044 = [0.06720075528287721,0.06458598051702369,0.026751517596986005,0.4138661709300592,0.3300653725315833,0.34003346027930403,0.2612432338137053,0.42177644482814225,0.09917753126022559,
0.4810902878839749,0.36727784110385114,0.3330512147470067,0.4076051649649473,0.1320779459126978,0.35529096282472156,0.008218244298513727,0.2946229537533901,0.43702871237416036,0.3902032177360908,0.4335142735944787,0.15632255339172574,0.03951934867982215,
0.48324200919216775,0.42898856909805005,0.17997928063630736,0.016441478126753346,0.36766841321115784,0.35326964445047254,0.29751858005926324,0.007846533452348453]

# RUIDO EN LOS PESOS DE LAS CONEXIONES
# # Agente 1
# fitness0 = [0.27, 0.2668, 0.176, 0.2356, 0.2732, 0.20, 0.16444, 0.2329, 0.284348, 0.2582887, 0.2310, 0.18948, 0.284434, 0.2582, 0.28744028, 0.3146814, 0.24239, 0.3314446, 0.19412, 0.163741, 0.1621, 0.2598, 0.1889, 0.1705, 0.2367, 0.2505, 0.2038, 0.1910, 0.2352, 0.2715399]
# fitness02 = [0.10507140591681857,0.09122212266582735,0.1067733886749922,0.12496882392089749,0.11628086002689433,0.10007638124708707,0.10876199304480458,0.11643396995733413,0.12166337062073555,0.14131750396544102,0.09715752776738969,0.105621954070167,0.12039134059625811,0.1109133404570537,0.1250147698041681,0.13216938727970845,0.1025368648324293,0.11475271533377632,0.11086059438851864,0.10686037223818809,0.12409091695099454,0.13684216234343483,0.14382693794216783,0.11005087438930714,0.10854043910915502,0.09706635607809916,0.11926177770902154,0.10636864651631422,0.137634772678426,0.10760472766039426]
# fitness034 = [0.11244785243292822,0.12335959614887024,0.11481565098161717,0.12896206713988456,0.1443277028920431,0.11894182223140809,0.10452808947331493,0.15141584729366264,0.10106023707582537,0.1141646439073427,0.1057601794048978,0.09868261658328355,0.13096937488556298,0.1031782716174099,0.11396140982722228,0.12224980171986444,0.12587193127780724,0.10652925536920536,0.1714034755825996,0.13306190319740024,0.1457074241840249,0.10603706973514113,0.1011372016458746,0.11408604459451016,0.10006107547653785,0.10296172508746587,0.10256991500338726,0.10753818275449083,0.09916006349462347,0.09875088086624137]
# fitness044 = [0.1481960275862769,0.12198909157809505,0.19445845018043556,0.10123859085654163,0.1652227502865489,0.08960783663308658,0.0966361480140353,0.11460196188364755,0.19671432188386823,0.1855365082146169,0.09784928289121689,0.14074024652401806,0.15843469080722408,0.1303935990764174,0.1162902503384116,0.1364460551391145,0.12709625258477614,0.14608832246624703,0.1263598467395277,0.11084482211516719,0.11512701160421684,0.12733890055287617,0.12015450840928218,0.13405603488194062,0.1096005715899031,0.10307953952947176,0.12433546960901701,0.12219990670007669,0.1454647374463558,0.11481585499375195]

# Agente 2
# fitness0 = [0.32, 0.459, 0.37, 0.33, 0.409, 0.274, 0.4667, 0.3879, 0.4511, 0.40, 0.458, 0.4926, 0.488, 0.293, 0.327, 0.448, 0.302, 0.3968, 0.383, 0.386, 0.36, 0.32, 0.38, 0.374, 0.371, 0.3418, 0.358, 0.348, 0.5055298, 0.35238]
# fitness02 = [0.12215255750701284,0.11268323839917385,0.1184260894072873,0.13799965207805184,0.12146197586608215,0.11938810486961379,0.14646218126772367,0.11475818272985962,0.11365519995822354,0.11672911741532832,0.10980834849809726,0.09689119089739229,0.1326576606151864,0.11237755540021442,0.11805924600062596,0.10579557077489628,0.10253638350229619,0.13638676681636552,0.10352331870764202,0.12101422607428859,0.12546376243727556,0.13208681409488177,0.1040686446572294,0.12123733191993749,0.11372748581448562,0.10335872450746034,0.1355800521235953,0.12033692451916349,0.11779468737285437,0.11528553102039468]
# fitness034 = [0.1530122536411906,0.13072557954994,0.13974809597528007,0.12534390746565455,0.10947261933681637,0.13199756191763723,0.12356349130087504,0.10310406750682138,0.13235710028764805,0.10388945902142602,0.13430987993690935,0.11775207678197419,0.1443201924194993,0.12386446822838562,0.13919013430412785,0.10463604713501859,0.14011498702264608,0.12324340737126574,0.11829146553865535,0.11513586134759778,0.1437296477689471,0.1256032670366522,0.11653402878250045,0.13903994151083687,0.13029887530271245,0.14177624967125757,0.11131989521495973,0.10973763408926981,0.1066242595846592,0.12400138121024903]
# fitness044 = [0.12524193386445331,0.13845069638580765,0.141260433683879,0.12388298299583464,0.14320875065203234,0.13546040552167904,0.11677159418973813,0.15698381233872777,0.13127769911667153,0.11718396995178519,0.13467155741046738,0.14795275723019422,0.12383549759002276,0.1304478913866063,0.11961624982544733,0.13234297498728595,0.11782188862701659,0.1440286480348711,0.11071716593114726,0.15327047264653082,0.17043455970687715,0.13604922836063355,0.12174011756757507,0.13056460027717004,0.12690019455890256,0.12478141275772055,0.12324428975520982,0.1401352143198213,0.15387926326937332,0.14436318272296067]

if __name__ == "__main__":
    data = [fitness0, fitness02, fitness034, fitness044]
    plt.boxplot(data)

    # print(len(fitness0))
    # print(len(fitness02))
    # print(len(fitness034))
    # print(len(fitness044))


    # plt.show()

    print(stats.ttest_ind(fitness0,fitness034, equal_var = True))
