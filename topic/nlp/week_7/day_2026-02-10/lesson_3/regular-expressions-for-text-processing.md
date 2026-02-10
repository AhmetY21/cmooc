---
title: "Regular Expressions for Text Processing"
date: "2026-02-10"
week: 7
lesson: 3
slug: "regular-expressions-for-text-processing"
---

# Topic: Regular Expressions for Text Processing

## 1) Formal definition (what is it, and how can we use it?)

A regular expression (regex or regexp) is a sequence of characters that define a search pattern. It's a powerful tool used for matching, locating, and manipulating text based on these patterns. They are essentially a mini-language for describing text structures.

We can use regular expressions for several purposes:

*   **Searching:** Finding specific patterns within a large body of text. For example, searching for all email addresses in a document.
*   **Validating:** Verifying if a string conforms to a specific format. For example, validating a phone number or an email address.
*   **Replacing:** Substituting occurrences of a pattern with another string. For example, standardizing date formats in a text file.
*   **Splitting:** Dividing a string into multiple substrings based on a pattern. For example, splitting a sentence into individual words.
*   **Extracting:** Isolating specific parts of a string that match a pattern. For example, extracting the area code from a phone number.

The power of regular expressions lies in their ability to define complex patterns using special characters and operators. These characters, called metacharacters, have specific meanings that allow you to represent various aspects of text structure, such as character ranges, repetitions, and alternatives. Examples include `.` (any character), `*` (zero or more occurrences), `+` (one or more occurrences), `?` (zero or one occurrences), `[]` (character class), and `\d` (digit). By combining these metacharacters, you can create sophisticated search patterns to precisely target the text you need.

## 2) Application scenario

Consider a scenario where you need to extract all email addresses from a large text file containing customer reviews. This file contains various text, including customer names, reviews, and their contact information. Manually searching for email addresses would be time-consuming and error-prone.

Using a regular expression, you can efficiently locate all email addresses within the text file. A suitable regular expression for this purpose might be:

`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`

This pattern looks for:

*   One or more alphanumeric characters, periods, underscores, percent signs, plus or minus signs (`[a-zA-Z0-9._%+-]+`) before the `@` symbol.
*   The `@` symbol itself.
*   One or more alphanumeric characters or periods (`[a-zA-Z0-9.-]+`) after the `@` symbol.
*   A period (`\.`).
*   Two or more alphabetic characters (`[a-zA-Z]{2,}`) representing the top-level domain (e.g., com, org, net).

This regex, when applied to the text file, would identify and extract all strings that match the pattern, effectively extracting all the email addresses.  This could then be used to contact customers, perform sentiment analysis associated with specific emails or for other marketing activities.

## 3) Python method

The `re` module in Python provides support for regular expressions. Here's how you can use it to extract email addresses from a text file:

```python
import re

def extract_emails(text):
  """
  Extracts all email addresses from a given text.

  Args:
    text: The input text string.

  Returns:
    A list of email addresses found in the text.
  """
  pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
  emails = re.findall(pattern, text)
  return emails

# Example Usage
text = "Contact us at support@example.com or sales.info@company.net for assistance.  John.Doe@subdomain.example.org is also available."
extracted_emails = extract_emails(text)
print(extracted_emails)
# Output: ['support@example.com', 'sales.info@company.net', 'John.Doe@subdomain.example.org']

# Example Reading from a File
try:
    with open("customer_reviews.txt", "r") as file:
        file_content = file.read()
        emails = extract_emails(file_content)
        print("Emails from file:", emails)
except FileNotFoundError:
    print("File not found: customer_reviews.txt")
```

Key elements of the code:

*   `import re`: Imports the regular expression module.
*   `re.findall(pattern, text)`: This function searches the text for all non-overlapping occurrences of the `pattern` and returns them as a list.
*   `r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"`: The `r` prefix before the string indicates a raw string, which prevents backslashes from being interpreted as escape sequences.  This is important for regex because backslashes are used extensively in the pattern itself.

## 4) Follow-up question

How would you modify the regular expression to only extract email addresses from the `.com` domain?  How could this be achieved programmatically without altering the regex itself, by filtering the results of the original regex? What are the advantages and disadvantages of each approach?