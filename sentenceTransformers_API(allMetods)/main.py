from sentence_transformers import SentenceTransformer, util
import torch
import pickle

# create the embedder instance
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print(embedder)


# the corpus below
corpus = ["There are so many method options! To help you remember how some of them are similar, we've grouped them into three categories",
"Infrequent Use methods are methods which are long acting, once you get them, they work on their own without you having to do anything. Examples include IUDs, implants, and sterilization.",
"IUD is a non-hormonal contraceptive method that has fewer side effects, requires no daily upkeep and protects us for up to 10 years. It can be removed at any time if you want to get pregnant.",
"the IUD stays in the womb, just like a baby doesn't move around during pregnancy.",
"the IUD is inside the womb, just like a man can't feel a baby during pregnancy. The only part that might be felt sometimes are the soft strings in the vagina, which are not painful.",
"the IUD has been used for many years in many countries and has not been shown to cause cancer. Many women will develop fibroids in their life and there is not a known cause, but using an IUD is not associated with having fibroids.",
"Studies show that women who have had IUDs removed have the same chance of getting pregnant as women who have never used contraception.",
"Pregnancy while using the IUD is very rare, and in this case you should have the IUD removed when you find out you are pregnant. Even if it is not removed, the IUD will be outside of the sac that the baby develops in so cannot touch the baby.",
"Implant is a small stick or soft capsule that is placed just under the skin on the inner arm. When placed, the implant is invisible and provides 3 to 5 years of protection against pregnancy. It can be removed at any time if you want to get pregnant.",
"the Implant stays in the arm, right below the skin. You can usually feel it by pressing on the area it was inserted. If you gain weight in your arm, sometimes you cannot feel it, but it can still be found with an x-ray.",
"Other people could feel the implant in your arm if they knew where it is located, but no one can know just by looking at you.",
"Implants have not been shown to cause cancer and might even protect against some forms of cancer. Many women will develop fibroids in their life and there is not a known cause, but using an implant is not associated with having fibroids.",
"Studies show that women who have had implants removed have the same chance of getting pregnant as women who have never used contraception.",
"you can have the implant removed at any time for any reason and you will be able to get pregnant right away.",
"it is normal, safe and healthy to not have a period when using the implant, like how women who are breastfeeding do not have periods. This can help prevent anemia. Your period will return once you have the implant removed",
"Tubal ligation is surgery that is done the same day. It is permanent and irreversible. Its purpose is to 'close' the tubes to prevent eggs and sperm from accessing the tubes.",
"Vasectomy is a same-day surgical procedure. It is permanent and irreversible. The goal is to block the passage of sperm to the urethra by blocking the vas deferens. Semen still contains as much seminal fluid, but less sperm.",
"There is no major change in period patterns after female sterilization. However, a woman's monthly periods usually becomes less regular as she approaches menopause.",
"Sterilization is intended to be permanent. People who want to have more children should choose a different method of contraception.",
"After sterilization, a woman will look and feel the same as before. She can have sex like before. She may find that she enjoys sex more because she doesn't have to worry about getting pregnant.",
"In order to do the tubal ligation, women are given local anesthesia to stop the pain and they stay awake. A woman can feel the provider move her uterus and fallopian tubes. It can be uncomfortable. A woman may have pain and feel weak for several days or even weeks after surgery, but she will soon regain her strength.",
"Frequent Use methods are methods which need to be taken or re-administered regularly. Examples include oral contraceptive pills, injectables, and cycle beads.",
"The pill is a small tablet that should be taken at the same time every day. The pill works using different hormones. There are pills containing one or two hormones, depending on your needs.",
"You should take the pill every day at around the same time. As long as you take it once a day, you are protected from pregnancy.",
"The pill protects against some forms of cancer. There is a temporary increased risk of breast cancer that goes away with time. Many women will develop fibroids in their life and there is not a known cause, but using the pill is not associated with having fibroids.",
"Former pill users have the same chance of getting pregnant as non-users.",
"Injectable is a hormonal contraceptive which is administered by injection every 2 or 3 months by a health professional but there is a range allowing self-injection based on the guidance of a provider.",
"No one will know you are using the injectable unless you tell them. If you experience bleeding changes, a partner may notice.",
"The injectable has not been shown to cause cancer and might even protect against some forms of cancer. Many women will develop fibroids in their life and there is not a known cause, but using the injectable is not associated with having fibroids.",
"For some women, it can take a few extra months for the injectable to stop working, but former injectable users have the same chance of getting pregnant at non-users.",
"It is normal, safe and healthy to not have a period when using the injectable, like how women who are breastfeeding do not have periods. This can help prevent anemia. Your period will return once you stop using the injectable.",
"Cycle beads It is a necklace of colored beads that represent each day of the woman's menstrual cycle (suitable for women who have cycles of 26 to 32 days). It helps you know when you can get pregnant. White beads mark the days when you can get pregnant. Brown beads mark the days when you're unlikely to get pregnant",
"Couples need to be highly motivated, well trained in the method, and determined to avoid unprotected sex during the fertile window. Some couples abstain from sex during the fertile window, others use another method during that time, such as a condom.",
"The number of days for abstinence  varies depending on the length of the woman's cycle. The average number of days a woman would be considered fertile and would need to abstain or use another method varies between 12 and 18 days depending on the methodology for monitoring fertility. To avoid unwanted pregnancy, you should use another method of birth control, such as condoms, or abstain from sex, during your fertile days.",
"During menstruation, the chances of pregnancy are low but not zero. The period itself does not prevent pregnancy, nor does it promote pregnancy. During the first days of the monthly period, the chances of pregnancy are lowest. As the days go by, the chances of pregnancy increase whether or not she is still bleeding. The risk of pregnancy increases until ovulation.",
"On Demand methods are methods that you use only when the need arises. Examples include condoms and emergency contraception.",
"EC(emergency contraceptive) is the morning-after pill containing progestin that prevents ovulation or delays ovulation. But if this ovulation has already taken place, emergency contraception is no longer useful.",
"EC only works to prevent pregnancy, and will not work if you are already pregnant, which is why the sooner after sex you take it, the more likely it is to work",
"EC may be taken often, even during the same menstrual cycle. However, EC is not as effective as other methods, so if you are using it regularly you are at a higher risk for pregnancy."
"EC leaves the body very quickly and does not have long-term side effects."
"Research shows that women who use EC have the same chance of getting pregnant as women who have never used contraception.",
"EC should be used as soon as possible after unprotected sex. It can work up to 5 days after sex, but the sooner you take it the better it works.",
"EC is not as effective as other methods, so if you are using it regularly you are at a higher risk for pregnancy.",
"If you vomit within 2 hours of taking EC, take another dose.",
"A condom is a thin latex or polyurethane sheath that is placed on the erect penis (male condom) or in the vagina (female condom) before intercourse and allows you to prevent both an unwanted pregnancy and of STIs / HIV.",
"Condom is the only method that prevents HIV and STIs as well as pregnancy",
"You can use one condom at a time, and only once. Make sure it is not expired and the packaging is not damaged.",
"Condoms can be used by sexually active men (using a male condom that covers the penis after the penis is erect) or women (using a female condom, that is slightly larger than a male condom and that is inserted into the vagina before sex), and there are many types. You can get them at a health facility, pharmacy, or store.",]

# embed corpus content
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)


print(corpus_embeddings)

def serializeStuff(object, filename):
    filename = filename+".pickle"
    with open(filename, 'wb') as f:
        pickle.dump(object, f)
    return "Done"

# # serialize the corpus
# result = serializeStuff(corpus,"corpus")
# print(result)


# serialize the corpus embeddings
result = serializeStuff(corpus_embeddings,"corpus_embeddings")
print(result)

# serialize the embedder
result = serializeStuff(embedder,"embedder")
print(result)