import numpy
import random

# Assumes the average user has taken 5 courses with some variation
# Should really analyze distribution of empirical data to determine these values
COURSE_COUNTS_PER_USER = [3, 4, 5, 6, 7]
COURSE_PROBABILITIES_PER_USER = [0.1, 0.2, 0.4, 0.2, 0.1]

# Assumes there are 60 relevant CS classes total
NUM_COURSES = 60

# There are 18 preference attributes
NUM_PREFERENCES = 18


if __name__ == "__main__":
    file = open("test_data.csv", "w")

    # added new line to make sure first row is not on attribute row
    file.write("userid,isCourseRating,attributeId,rating\n")

    # generate data for 10,000 users
    for user_id in range(10000):

        # add preference ratings
        for pref_id in range(NUM_PREFERENCES):
            rating = random.randint(1,5)
            file.write("%d,0,%d,%d\n" % (user_id, pref_id, rating))

        # add course ratings
        num_courses_per_user = numpy.random.choice(COURSE_COUNTS_PER_USER, p=COURSE_PROBABILITIES_PER_USER)
        course_ids = numpy.random.choice(NUM_COURSES, size=num_courses_per_user, replace=False)
        for course_id in course_ids:
            rating = random.randint(1, 5)
            file.write("%d,1,%d,%d\n" % (user_id, course_id, rating))

    file.close()

