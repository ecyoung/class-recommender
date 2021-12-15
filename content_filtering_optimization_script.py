# This script optimized hyperparameters for content_filtering by minimizing RMSE

import content_filtering
from input_parsing import parseInputData
import knn_util
import numpy
import math


def getTotalRMSE(kNN, users, courses, k, type_of_class_weight, topic_weight, knn_weight):
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
        predicted_ratings = content_filtering.getPredictedCourseRatingsApproach1_opt((kNN[user_id])[:k], courses, users, user_id, True, type_of_class_weight, topic_weight, knn_weight)
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


def findOptimalValuesApproach1(courses, users):
    userList = users.keys()

    kNN = dict()

    best_combo = (0, 0, 0, 0, 0)
    minRMSE = float("inf")
    outer_weightCombinations = [(0, 1), (0.05, 0.95), (0.1, 0.9), (0.15, 0.85), (0.2, 0.8), (0.25, 0.75), (0.3, 0.7), (0.35, 0.65),
                          (0.4, 0.6), (0.45, 0.55), (0.5, 0.5), (0.55, 0.45), (0.6, 0.4), (0.65, 0.35), (0.7, 0.3),
                          (0.75, 0.25), (0.8, 0.2), (0.85, 0.15), (0.9, 0.1), (0.95, 0.05), (1, 0)]
    inner_weightCombinations = []

    knn_weights = numpy.linspace(0, 1, 21)
    for weight in knn_weights:
        type_of_class_weights = numpy.linspace(0, 1 - round(weight, 2), int((1 - round(weight, 2)) / 0.05) + 1)
        for type_of_class_weight in type_of_class_weights:
            topic_weight = 1 - round(type_of_class_weight, 2) - round(weight, 2)
            inner_weightCombinations.append((round(weight, 2), round(type_of_class_weight, 2), round(topic_weight, 2)))

    for pref_weight, course_weight in outer_weightCombinations:
        for user_id in userList:
            kNN[user_id] = knn_util.knn_opt(users, user_id, 1, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}, pref_weight, course_weight)
        for k in range(1, 41):
            for knn_weight, type_of_class_weight, topic_weight in inner_weightCombinations:
                RMSE = getTotalRMSE(kNN, users, courses, k, type_of_class_weight, topic_weight, knn_weight)
                if RMSE < minRMSE:
                    minRMSE = RMSE
                    best_combo = (k, pref_weight, course_weight, knn_weight, type_of_class_weight, topic_weight)

    return best_combo, minRMSE


if __name__ == "__main__":
    (users, courses) = parseInputData("survey_responses.csv", "course_tags.csv")
    comparatorSet = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}

    result = findOptimalValuesApproach1(courses, users)
    print("Optimal settings: ")
    print("k: " + str((result[0])[0]))
    print("pref_weight: " + str((result[0])[1]))
    print("course_weight: " + str((result[0])[2]))
    print("knn_weight: " + str((result[0])[3]))
    print("type_of_class_weight: " + str((result[0])[4]))
    print("topic_weight: " + str((result[0])[5]))
    print("RMSE: " + str(result[1]))
