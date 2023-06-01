# AWS Instructions

## Getting Started
### Configuring AWS Credentials (first-time only)
1. Install [AWS CLI.](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
2. Run the following command:
    ```bash
    aws configure sso
    ```
3. Obtain the appropriate `SSO Start URL` from the access portal after logging in to AWS by clicking on "Command line or programmatic access".
4. Follow the instructions given by the interactive CLI Login Wizard to configure your SSO profile.
    - Make sure to set the SSO region to `us-east-2`.
    - The profile name you set and the end of the configuration will be used to SSO into AWS from now on.

### Using AWS Credentials
1. Use the following command to connect to AWS through SSO with the your credentials:
    ```bash
    aws sso login --profile <your-named-profile>
    ```
2. Finally, export your AWS credentials securely as environment variables:
    ```bash
    eval $(aws configure export-credentials --format env --profile <profile-name>)
    ```

### Starting up the EC2 Instance
1. First, SSH into the appropriate EC2 instance:
    ```bash
    ssh -i <key-of-EC2.pem> ubuntu@<IP-of-EC2>
    ```
2. Activate the corresonding Python virtual environment:
    ```bash
    source <virtual-environment-name>/bin/activate
    ```
3. Run the following command to start hosting the model end point:
    ```bash
    python predict_activity.py
    ```

### Connecting to WebSocket
1. Install [npm.](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm/)
2. Install [wscat.](https://www.npmjs.com/package/wscat)
3. Run the following command:
    ```bash
    wscat -c <WebSocketURL>
    ```
## Running the Solution
### Stream Data using KDS
1. CD into the `/KDS` directory.
    ```bash
    cd KDS
    ```
2. Stream the data for any of the use cases after completing the necessary setup to run the automated end-to-end solution on AWS: 
    ```bash
    python <use-case>_KDS.py
    ```
