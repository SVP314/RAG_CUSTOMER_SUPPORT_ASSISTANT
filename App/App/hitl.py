def escalate_to_human(state):
    human_response = input("Enter human support response: ")
    state["human_response"] = human_response
    state["answer"] = human_response
    state["escalated"] = True
    return state
