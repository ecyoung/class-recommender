from class_recommender import *

import time
import matplotlib.pyplot as plt


def populate_users(data, num_users, num_attributes):
    users = {}
    data.readline()
    for line in data:
        line = line.split(',')
        userid = int(line[0])
        isCourse = bool(line[1])
        attributeid = int(line[2])
        rating = int(line[3])

        if userid < num_users and (isCourse or attributeid < num_attributes):
            if userid not in users:
                users[userid] = User(userid, {}, {})

            if isCourse:
                users[userid].add_course_rating(attributeid, rating)
            else:
                users[userid].add_preference(attributeid, rating)

    return users


if __name__ == "__main__":
    output = open("runtime_results.csv", "w")
    for i in range(18, 0, -1):
        times = []
        output.write("%d," % i)
        for N in range(1000, 10001, 1000):
            users = {}
            data = open("test_data.csv", "r")
            users = populate_users(data, N, i)

            start = time.time()
            results = getPredictedCourseRatings(users, 0, 30, 1, range(i), 5, 3, False)
            end = time.time()
            print("Runtime for i=%d, N=%d: %.03f" % (i, N, end-start))
            times.append(end-start)
            output.write("%.06f," % (end-start))
            data.close()
        output.write("\n")
        plt.plot(range(1000, 10001, 1000), times, label="i="+str(i))
    output.close()
    plt.ylabel('Runtime (seconds)')
    plt.xlabel('Number of users')
    plt.title("Class Recommender runtimes for varying preference attributes count (i)", wrap=True)
    plt.legend(bbox_to_anchor=(1.04, -0.04), loc="lower left")
    plt.savefig("figures/runtime_analysis", bbox_inches="tight")
    plt.show()




