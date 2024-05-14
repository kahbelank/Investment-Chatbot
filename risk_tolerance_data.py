# risk_tolerance_data.py

RISK_QUESTIONS = [
    "1. In general, how would your best friend describe you as a risk taker?",
    "2. You are on a TV game show and can choose one of the following; which would you take?",
    "3. You have just finished saving for a 'once-in-a-lifetime' vacation. Three weeks before you plan to leave, you lose your job. You would:",
    "4. If you unexpectedly received $20,000 to invest, what would you do?",
    "5. In terms of experience, how comfortable are you investing in stocks or stock mutual funds?",
    "6. When you think of the word 'risk,' which of the following words comes to mind first?",
    "7. Some experts are predicting prices of assets such as gold, jewels, collectibles, and real estate (hard assets) to increase in value; bond prices may fall, however, experts tend to agree that government bonds are relatively safe. Most of your investment assets are now in high-interest government bonds. What would you do?",
    "8. Given the best and worst case returns of the four investment choices below, which would you prefer?",
    "9. In addition to whatever you own, you have been given $1,000. You are now asked to choose between:",
    "10. In addition to whatever you own, you have been given $2,000. You are now asked to choose between:",
    "11. Suppose a relative left you an inheritance of $100,000, stipulating in the will that you invest ALL the money in ONE of the following choices. Which one would you select?",
    "12. If you had to invest $20,000, which of the following investment choices would you find most appealing?",
    "13. Your trusted friend and neighbor, an experienced geologist, is putting together a group of investors to fund an exploratory gold mining venture. The venture could pay back 50 to 100 times the investment if successful. If the mine is a bust, the entire investment is worthless. Your friend estimates the chance of success is only 20%. If you had the money, how much would you invest?",
]

OPTIONS = [
    ["a. A real gambler", "b. Willing to take risks after completing adequate research", "c. Cautious", "d. A real risk avoider"],
    ["a. $1,000 in cash", "b. A 50% chance at winning $5,000", "c. A 25% chance at winning $10,000", "d. A 5% chance at winning $100,000"],
    ["a. Cancel the vacation", "b. Take a much more modest vacation", "c. Go as scheduled, reasoning that you need the time to prepare for a job search", "d. Extend your vacation, because this might be your last chance to go first-class"],
    ["a. Deposit it in a bank account, money market account, or insured CD", "b. Invest it in safe high-quality bonds or bond mutual funds", "c. Invest it in stocks or stock mutual funds"],
    ["a. Not at all comfortable", "b. Somewhat comfortable", "c. Very Comfortable"],
    ["a. Loss", "b. Uncertainty", "c. Opportunity", "d. Thrill"],
    ["a. Hold the bonds", "b. Sell the bonds, put half the proceeds into money market accounts, and the other half into hard assets", "c. Sell the bonds and put the total proceeds into hard assets", "d. Sell the bonds, put all the money into hard assets, and borrow additional money to buy more"],
    ["a. $200 gain best case; $0 gain/loss worst case", "b. $800 gain best case, $200 loss worst case", "c. $2,600 gain best case, $800 loss worst case", "d. $4,800 gain best case, $2,400 loss worst case"],
    ["a. A sure gain of $500", "b. A 50% chance to gain $1,000 and a 50% chance to gain nothing"],
    ["a. A sure loss of $500", "b. A 50% chance to lose $1,000 and a 50% chance to lose nothing"],
    ["a. A savings account or money market mutual fund", "b. A mutual fund that owns stocks and bonds", "c. A portfolio of 15 common stocks", "d. Commodities like gold, silver, and oil"],
    ["a. 60% in low-risk investments, 30% in medium-risk investments, 10% in high-risk investments", "b. 30% in low-risk investments, 40% in medium-risk investments, 30% in high-risk investments", "c. 10% in low-risk investments, 40% in medium-risk investments, 50% in high-risk investments"],
    ["a. Nothing", "b. One month’s salary", "c. Three month’s salary", "d. Six month’s salary"],
]

SCORES = [
    [4, 3, 2, 1], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3],
    [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
    [1, 3], [1, 3], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]
]
