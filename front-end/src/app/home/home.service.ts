import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Credentials, CredentialsService } from '@app/auth';
import { environment } from '@env/environment';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class HomeService {
  constructor(
    private httpClient: HttpClient,
    private credentiasService: CredentialsService
  ) { 
    
  }

  createUser(userObj: any): Observable<any> {
    let url = 'newuser/';
    url = this.buildUrl(url, userObj);
    return this.httpClient.post(url, {})
  }

  createAccount(accountObj: any): Observable<any> {
    // http://localhost:9080/newaccount/?userid=2&bankid=1&cif=345345343&acc_type=1

    let url = 'newaccount/';
    url = this.buildUrl(url, accountObj)
    return this.httpClient.post(url, {});
  }

  getAccountSummary(accountObj: any): Observable<any> {
    // http://localhost:9080/accsummary/?acc_id=2
    let url = 'accsummary/';
    url = this.buildUrl(url, accountObj);
    return this.httpClient.get(url, {});
  }

  getTransactions(accountId: string): Observable<any> {
    // http://localhost:9080/trandetails/?acc_id=1
    let url = 'trandetails/?acc_id=' + accountId;
    return this.httpClient.get(url, {});
  }

  getbalance(acc_id: string): Observable<any> {
    // http://localhost:9080/getbalance/?acc_id=2
    const url = 'getbalance/?acc_id=' + acc_id;
    return this.httpClient.get(url, {});
  }

  checkForFraudulent(transferObj: any): Observable<any> {
    let url = 'http://localhost:8000/transaction/validate/'
    console.log("Transaction Object:", transferObj);

    transferObj = {
        "transaction_id": 1,
        "tran_type_id": 1,
        "transaction_amount": 50,
        "tran_date": "2025-06-03",
        "status_type_id": 1,
        "from_acc_id": 7,
        "to_acc_id": 5
    }

    const transaction = {
      from_acc_id : "4",
      to_acc_id : "2",
      transaction_amount : 100
    }

    // url = this.buildUrl(url, transferObj);
    return this.httpClient.post(url, transferObj);
  }

  sendSMS(): Observable<any> {
    let url = "http://localhost:9080/send-sms/"
    let SMSObj = {
      "to": "+919986730233",
      "message": "We have noticed a fraudulent transaction on your Neo Bank account. If you are the one making the transaction, please log in to approve it"
    }

    return this.httpClient.post(url, SMSObj)
  }

  transferAmount(transferObj: any): Observable<any> {
    // http://localhost:9080/transfer/?from_acc_id=2&to_acc_id=1&transaction_amount=2000
    let url = 'transfer/';
    url = this.buildUrl(url, transferObj);
    return this.httpClient.get(url, {});
  }

  loginUser(loginDetails: any): Observable<any> {
    // let url = 'http://localhost:9080/validatepass/?email_id=mithun.hallikere@gmail.com&passwd=futurebank';
    // let header_node = {
    //   headers: new HttpHeaders(
    //       { 'rejectUnauthorized': 'false' })
    //   };

    let url = 'validatepass/'
    // url = this.buildUrl(url, loginDetails);
    return this.httpClient.post(url, loginDetails);
  }

  getUserDetails(userObj: any): Observable<any> {
    // http://localhost:9080/getuserdetails/?user_id=1
    let url = 'getuserdetails/';
    url = this.buildUrl(url, userObj);
    return this.httpClient.get(url, {});
  }

  getAccounts(): Observable<any> {
    const credentials  = this.credentiasService.credentials;
    let url = 'getaccounts/?user_id=' + credentials?.userId;
    return this.httpClient.get(url, {});
  }

  buildUrl(url: string, dataObj: any) {
    if (dataObj) {
      url = url + '?';
      let keys = Object.keys(dataObj) || [];

      keys.forEach(key => {
        url = url + key + '=' + dataObj[key] + '&';
      })

      url = url.substring(0, url.length - 1);
      return url;

    } else {
      return url;
    }
  }

  generateCif() {
    return Math.floor(Math.random() * 10000000000);
  }
}
