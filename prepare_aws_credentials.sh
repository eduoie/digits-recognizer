export AWS_ACCESS_KEY_ID=`cat ${HOME}/.aws/credentials | grep aws_access_key_id | cut -f2 -d"=" | cut -f2 -d" "`
export AWS_SECRET_ACCESS_KEY=`cat ${HOME}/.aws/credentials | grep aws_secret_access_key | cut -f2 -d"=" | cut -f2 -d" "`
export AWS_DEFAULT_REGION=`cat ${HOME}/.aws/config | grep region | cut -f2 -d"=" | cut -f2 -d" "`
