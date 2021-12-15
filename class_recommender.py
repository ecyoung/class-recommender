import content_filtering
import input_parsing
import collaborative_filtering
import matrix_factorization


class Course:
    """
    Course class, storing course information
    
    :param: course.id = course id (int)
                        course_id's range from [18 to 65]
    :param: course.name = course name (string)

    :property: course.type_of_class_tags = list of types that correspond to the course
    :property: course.topic_tags = list of topics that correspond to the course
    :property: course.ratings = list of ratings for the course
    :property: course.average_rating = average rating for course (float)
    """
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.type_of_class_tags = []
        self.topic_tags = []
        self.ratings = []
        self.average_rating = 0


class User:
    """
    User class, storing ratings information of a user

    :param: user.id = user id (int)
    :param: preferences = dictionary mapping preference id to rating
                            preference_id's range from [0 to 17]

    :param: courseRatings = dictionary mapping course id to rating
                            course_id's range from [18 to 65]

    :property factorizedCourseRatings = dictionary mapping course id to factorized course rating
                                        course_id's range from [18 to 65]
    """

    def __init__(self, id, preferences, courseRatings):
        self.id = id
        self.preferences = preferences
        self.courseRatings = courseRatings
        self.factorizedCourseRatings = {}

    def add_preference(self, preference, rating):
        self.preferences[preference] = rating

    def add_course_rating(self, course, rating):
        self.courseRatings[course] = rating

def printPredictedCourseRatings(courses, predicted_ratings):
    """
    Prints formatted predicted course ratings as:
        course_name : predicted_rating
    """
    i = 1
    for key, value in predicted_ratings:
        print(i, ". ", courses[key].name, ' : ', value)
        i += 1

if __name__ == "__main__":
    # simple test to test that data is parsed correctly and works with recommender system
    (users, courses) = input_parsing.parseInputData("survey_responses.csv", "course_tags.csv")
    comparatorSet = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
    standard_prediction = collaborative_filtering.getPredictedCourseRatings(users, courses, 1, comparatorSet, False)
    matrix_factorization_prediction = matrix_factorization.getPredictedCourseRatings(users, courses, 0, comparatorSet, False)
    approach1_prediction = content_filtering.getPredictedCourseRatingsApproach1(courses, users, 1, comparatorSet, False)
    approach2_prediction = content_filtering.getPredictedCourseRatingsApproach2(courses, users, 1)

    print("Collaborative-Filtering:")
    printPredictedCourseRatings(courses, standard_prediction)
    print("")
    print("Matrix-Factorization:")
    printPredictedCourseRatings(courses, matrix_factorization_prediction)
    print("")
    print("Content-Based-Filtering (Approach 1):")
    printPredictedCourseRatings(courses, approach1_prediction)
    print("")
    print("Content-Based-Filtering (Approach 2):")
    printPredictedCourseRatings(courses, approach2_prediction)
    print("")
