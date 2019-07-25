import os

num_jobs = 0
latest_job = 0
a = os.popen("oarstat | grep aboytsov").read()
running_jobs = list()
for l in a.split("\n"):
    js = l.split("job")
    if len(js) > 1:
        num_jobs += 1
        j = int(js[1][:4].strip())
        running_jobs.append(j)
        if j>latest_job:
            latest_job = j
print(len(running_jobs),running_jobs)
if len(running_jobs)==0:
    print("No jobs running. All complete?")
print(num_jobs, latest_job)
latest_job = max(latest_job, 0)
jobs_to_run = [i for i in range(latest_job + 1, latest_job + 1 + (80-num_jobs)) if i < 100]
print(len(jobs_to_run), jobs_to_run)
for i in jobs_to_run:
   os.system('oarsub -n job'+str(i)+' -l nodes=1,walltime=3 "./single_letter_A_job_script.sh '+str(i)+'"')
