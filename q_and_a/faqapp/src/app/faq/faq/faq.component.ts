import { Component, OnInit } from '@angular/core';
import { HttpClient, HttpHeaders, HttpResponse } from '@angular/common/http';
@Component({
  selector: 'app-faq',
  templateUrl: './faq.component.html',
  styleUrls: ['./faq.component.css']
})
export class FaqComponent implements OnInit {

  text : string = "" ;
  answer : string = null ;
  loading : boolean = false;

  constructor(private http : HttpClient) 
  {

  }

   post(body,url="http://localhost:4000/api"){
    const headers = new HttpHeaders({
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin':'*',
      'Access-Control-Allow-Methods': 'GET,POST,PATCH,DELETE,PUT,OPTIONS',
      'Access-Control-Allow-Headers' : 'Origin, Content-Type, X-Auth-Token, content-type'
    });
  
    let options = {
      headers: headers
    };
    return this.http.post(url, body , options).subscribe((object:any)=>{
      this.loading = false ;
      this.answer = object.response.answer;
      console.log('response => ',object,  this.answer );
    });
   }
  
  ngOnInit()
  {
console.log('init ',this.text);
  }

  ask()
  {
    console.log('ask => ',this.text);
    this.loading = true;
    this.post({query:this.text});
  }
}
