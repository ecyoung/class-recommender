import numpy as np
import math
import heapq
import time
from input_parsing import *

def getPreferenceDistance(user1, user2, comparatorSet):
    """
    Get the angular distance between two users based on their ratings on preferences of interest

    :param user1: User object of the first user
    :param user2: User object of the second user
    :param comparatorSet: Set of preference ids (int) corresponding to the preferences of interest

    :return: float, the distance between user 1 and user 2
    """
    # initialize variables
    dotProduct = 0  # numerator of angular distance
    squaredSum1 = 0  # sum of (user 1's rating)^2
    squaredSum2 = 0  # sum of (user 2's rating)^2
    # get the preference mapping ({preference_id: rating}) of the two users
    # unfiltered (contains irrelevant preferences not in the comparatorSet)
    preference1 = user1.preferences
    preference2 = user2.preferences
    # loop through preferences of interest and get dot product and squared sums
    for pref_id in comparatorSet:
        rating1 = preference1[pref_id]
        rating2 = preference2[pref_id]
        dotProduct += (rating1 * rating2)
        squaredSum1 += (rating1 ** 2)
        squaredSum2 += (rating2 ** 2)
    # return angular distance
    denominator = math.sqrt(squaredSum1 * squaredSum2)
    return 1 - (dotProduct / denominator)

def getCourseDistance(user1, user2, threshold):
    """
    Get the angular distance between two users based on their ratings on courses

    :param user1: User object of the first user
    :param user2: User object of the second user
    :param threshold: Commonality threshold of #overlapping courses, return 1 if threshold not reached

    :return: float, the angular distance between user 1 and user 2
    """
    # get the course rating mapping ({course_id: rating}) of the two users
    courseRatings1 = user1.courseRatings
    courseRatings2 = user2.courseRatings
    # find intersection (ids of courses that both users rate)
    commonCourseIds = set(courseRatings1.keys() & courseRatings2.keys())

    # return 1 if commonality threshold not reached
    if len(commonCourseIds) < threshold:
        return 1
    # initialize variables
    dotProduct = 0  # numerator of angular distance
    squaredSum1 = 0 # sum of (user 1's rating)^2
    squaredSum2 = 0 # sum of (user 2's rating)^2
    # loop through preferences of interest and get dot product and squared sums
    for course_id in commonCourseIds:
        rating1 = courseRatings1[course_id]
        rating2 = courseRatings2[course_id]
        dotProduct += (rating1*rating2)
        squaredSum1 += (rating1**2)
        squaredSum2 += (rating2**2)
    # return angular distance
    denominator = math.sqrt(squaredSum1*squaredSum2)
    # quick fix (since dot product numerator is also 0)
    if denominator == 0.0:
        return 0
    return 1 - dotProduct/denominator


def getFactorizedCourseDistance(user1, user2):
    """
    Get the angular distance between two users based on their ratings on courses

    :param user1: User object of the first user
    :param user2: User object of the second user

    :return: float, the angular distance between user 1 and user 2
    """
    # get the course rating mapping ({course_id: rating}) of the two users
    courseRatings1 = user1.factorizedCourseRatings
    courseRatings2 = user2.factorizedCourseRatings
    # find intersection (ids of courses that both users rate)
    # initialize variables
    dotProduct = 0  # numerator of angular distance
    squaredSum1 = 0  # sum of (user 1's rating)^2
    squaredSum2 = 0  # sum of (user 2's rating)^2
    # loop through preferences of interest and get dot product and squared sums
    for course_id in range(18, 86):
        rating1 = courseRatings1[course_id]
        rating2 = courseRatings2[course_id]
        dotProduct += (rating1 * rating2)
        squaredSum1 += (rating1 ** 2)
        squaredSum2 += (rating2 ** 2)
    denominator = math.sqrt(squaredSum1 * squaredSum2)
    if denominator == 0.0:
        return 0
    return 1 - dotProduct / denominator

# considering all ratings (including the ones predicted by factorization)
def knn(users, user_id, threshold, k, comparatorSet, pref_weight=0.5, course_weight=0.5):
    """
    Get the K nearest neighbors of a user

    :param: users: dictionary that maps user_id to User object, contains information of all users
    :param user_id: user id of the target user
    :param threshold: commonality threshold
    :param k: number of nearest neighbors
    :param comparatorSet: set of ids of target preferences

    :return: list of user_ids (int) of the K nearest neighbors
    """
    # initialize priority queue, which will store (distance, user_id) pair
    heap = []
    # if k is greater than the size of all users, return all user id
    if k > len(users):
        return users.keys()
    # compute the angular distances between the user and all other users
    for curr_id in users.keys():
        if user_id == curr_id:
            continue
        # get angular distances (takes in User object)

        # version 1 get knn with all ratings (actual + predicted)
        # preference_distance = getPreferenceDistance(users.get(user_id), users.get(curr_id), comparatorSet)
        # course_distance = getFactorizedCourseDistance(users.get(user_id), users.get(curr_id))
        # distance = preference_distance*pref_weight + course_distance*course_weight

        # version 2 get knn with preference ratings only
        # preference_distance = getPreferenceDistance(users.get(user_id), users.get(curr_id), comparatorSet)
        # distance = preference_distance

        # version 3 get knn with actual ratings
        preference_distance = getPreferenceDistance(users.get(user_id), users.get(curr_id), comparatorSet)
        course_distance = getCourseDistance(users.get(user_id), users.get(curr_id), threshold)
        distance = preference_distance*pref_weight + course_distance*course_weight

        # stores neighbor distances in the form of a tuple: (neighbor user_id, angular distance)
        heapq.heappush(heap, (distance, curr_id))

    knn = []
    for i in range(k):
        if len(heap) > 0:
            # append the user id to knn
            knn.append(heapq.heappop(heap)[1])
    return knn


def getPredictedCourseRatings(users, courses, user_id, comparatorSet, is_validation, r=5, k=20, threshold=1, pref_weight=0.2, course_weight=0.8):
    """
    Get the r top recommended courses for a user

    :param: users: dictionary that maps user_id to User object, contains information of all users
    :param courses: dictionary that maps course_id to Course object, contains information of all courses
    :param user_id: user id of the target user
    :param comparatorSet: set of ids of target preferences
    :param is_validation: whether the recommendation is for validation purposes or not

    :param r: number of recommendations to output
    :param k: number of nearest neighbors
    :param threshold: commonality threshold
    :param pref_weight: weight of preference similarity between users to determine overall similarity
    :param course_weight: weight of course similarity between users to determine overall similarity

    :return: dictionary mapping course_id to predicted_rating for that course id
    """
    kNNs = knn(users, user_id, threshold, k, comparatorSet, pref_weight, course_weight)
    # a dictionary mapping course_id to a list of all ratings by kNN
    ratings = {}
    # courses taken by the target user
    courses_taken = users.get(user_id).courseRatings.keys()

    if not is_validation:
        all_courses = set(courses.keys())
        courses_to_rate = all_courses.difference(courses_taken)
    else:
        courses_to_rate = courses_taken

    for course_id in courses_to_rate:
        # course_id's range from: [18=CS101 to 85=CS675D]
        if not isValidCourse(course_id):
            continue
        ratings[course_id] = []
        for neighbor_id in kNNs:
            ratings[course_id].append(users[neighbor_id].factorizedCourseRatings[course_id])

    # initialize priority queue, which will store (predicted_rating, course_id) pairs
    heap = []
    recommendations = []
    # for each course, predict the score and add it to the priority queue
    for course in courses_to_rate:
        if not isValidCourse(course_id):
            continue
        n = len(ratings[course])
        average = 0
        for rating in ratings[course]:
            average += rating / n
        predicted_rating = (courses[course].average_rating + (n * average)) / (1 + n)
        if is_validation:
            recommendations.append((course, predicted_rating))
        if not is_validation:
            heapq.heappush(heap, (-predicted_rating, course))

    if is_validation:
        return recommendations

    if r > len(heap):
        r = len(heap)

    for i in range(r):
        course = heapq.heappop(heap)
        recommendations.append((course[1], -course[0]))

    # return only top r courses in priority queue
    return recommendations


def isElective(course_id):
    """
    Returns whether or not the course_id is an elective or not
    """
    # course ids that are 28-85 are all electives
    return (course_id >= 28 and course_id <= 85)

def getTotalRMSE(users, courses, comparatorSet, k=20, threshold= 1, r=5):
    userList = users.keys()
    # initialize total terms
    totalTerms = 0
    # initialize sum of squared diffs term
    sumOfSquaredDiffs = 0
    for user_id in userList:
        # user is a User object that contains information about preferences and ratings
        user = users[user_id]
        # total number of course ratings for the user
        numRatings = len(user.courseRatings)
        predicted_ratings = getPredictedCourseRatings(users, courses, user_id, comparatorSet, True, r, k, threshold)
        for course_id, rating in predicted_ratings:
            actualRating = user.courseRatings[course_id]
            predictedRating = rating
            squaredDiff = (actualRating - predictedRating)**2
            # increment sumOfSquaredDiffs by the squaredDiff of current predicted vs actual rating
            sumOfSquaredDiffs += squaredDiff
        # increment total terms with number of ratings of user
        totalTerms += numRatings
    # divide sumOfSquaredDiffs by total terms to obtain average and take square root
    rmse = math.sqrt(sumOfSquaredDiffs / totalTerms)
    return rmse

def get_userItem_matrix(users, numCourses):
    matrix = [[0 for _ in range(numCourses)] for _ in range(len(users))]
    for user_id in users.keys():
        for course_id in users[user_id].courseRatings.keys():
            if course_id < numCourses:
                matrix[user_id][course_id] = users[user_id].courseRatings[course_id]
    return matrix

def matrix_factorization(R, K, steps=1000, alpha=0.0002, beta=0.02):
    """
    :param R:
    :param K:
    :param steps:
    :param alpha:
    :param beta:
    :return:
    """
    P = np.random.rand(len(R),K)
    Q = np.random.rand(len(R[0]), K)
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return np.dot(P, Q)

def normalize_matrix(data):
    """
    Rescale the matrix such that the values lie in between 1 and 5
    """
    return (((data - np.min(data))*5) / (np.max(data) - np.min(data)))+1\

def isValidCourse(id):
    """
    Returns whether the id is a valid course id.
    course_id's range from: (18 to 85) where 18=CS101 and 85=CS675D
    """
    return id >= 18 or id <= 85

if __name__ == "__main__":
    # functions modified: getPredictedCourseRatings, getTotalRMSE, parseSurveyData
    # functions added: getCompletedUsers, isValidCourse
    (users, courses) = parseInputData("survey_responses.csv", "course_tags.csv")

    # train factorization and write the matrix to csv file
    # matrix = get_userItem_matrix(users, 68)
    # completed_matrix = np.array(matrix_factorization(np.array(matrix), 30, steps=5000))
    # completed_matrix = normalize_matrix(completed_matrix)
    # output = open("complete_matrix2.csv", "w")
    # for row in completed_matrix:
    #     for rating in row:
    #         output.write("%.06f," % rating)
    #     output.write("\n")
    comparatorSet = set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    start = time.time()
    RMSE = getTotalRMSE(users, courses, comparatorSet)
    end = time.time()
    courses_pair = getPredictedCourseRatings(users, courses, 0, comparatorSet, False)
    print("RMSE: ", RMSE)
    print("time ", (end-start))